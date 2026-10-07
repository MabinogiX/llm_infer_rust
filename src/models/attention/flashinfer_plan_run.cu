#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

namespace {
using Variant = flashinfer::DefaultAttention<false, false, false, false>;

struct NativePlan {
  flashinfer::DecodePlanInfo info{};
  flashinfer::PrefillPlanInfo prefill_info{};
  void* float_workspace = nullptr;
  void* int_workspace = nullptr;
  void* pinned_int_workspace = nullptr;
  size_t float_capacity = 0;
  size_t int_capacity = 0;
  size_t pinned_capacity = 0;
  uint64_t allocations = 0;

  cudaError_t reserve(void*& buffer, size_t& capacity, size_t bytes, bool pinned) {
    if (bytes <= capacity) return cudaSuccess;
    size_t grown = std::max(bytes, capacity ? capacity * 2 : size_t(65536));
    void* replacement = nullptr;
    auto status = pinned ? cudaMallocHost(&replacement, grown)
                         : cudaMalloc(&replacement, grown);
    if (status != cudaSuccess) return status;
    if (buffer) {
      status = pinned ? cudaFreeHost(buffer) : cudaFree(buffer);
      if (status != cudaSuccess) {
        if (pinned) cudaFreeHost(replacement); else cudaFree(replacement);
        return status;
      }
    }
    buffer = replacement;
    capacity = grown;
    ++allocations;
    return cudaSuccess;
  }

  ~NativePlan() {
    if (float_workspace) cudaFree(float_workspace);
    if (int_workspace) cudaFree(int_workspace);
    if (pinned_int_workspace) cudaFreeHost(pinned_int_workspace);
  }
};

template <typename T, uint32_t GroupSize>
cudaError_t make_plan(NativePlan& plan, const int32_t* indptr_host,
                      uint32_t batch_size, uint32_t num_q_heads,
                      uint32_t page_size, bool graph, cudaStream_t stream) {
  using Params = flashinfer::BatchDecodeParams<T, T, T, int32_t>;
  auto estimate = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GroupSize, 128, flashinfer::PosEncodingMode::kNone, Variant, Params>;
  bool split_kv = false;
  uint32_t max_grid_size = 0, max_pages = 0, new_batch_size = 0, grid_y = 0;
  auto status = estimate(split_kv, max_grid_size, max_pages, new_batch_size,
                         grid_y, batch_size, const_cast<int32_t*>(indptr_host),
                         num_q_heads, page_size, graph, stream);
  if (status != cudaSuccess) return status;

  // FlashInfer's DecodePlan uses four aligned integer arrays and, when
  // split_kv is selected, temporary value and score arrays.
  const size_t padded = graph ? (split_kv ? max_grid_size / grid_y : batch_size)
      : std::max<size_t>(new_batch_size, batch_size);
  const size_t int_bytes = padded * 17 + 128;
  const size_t float_bytes = split_kv
      ? size_t(num_q_heads) * padded * (128 * sizeof(float) + sizeof(float)) + 64
      : 1;
  // A captured graph holds these addresses. Reject growth rather than silently
  // freeing a workspace referenced by its kernels.
  if (plan.info.enable_cuda_graph &&
      (float_bytes > plan.float_capacity || int_bytes > plan.int_capacity ||
       int_bytes > plan.pinned_capacity)) return cudaErrorInvalidValue;
  if ((status = plan.reserve(plan.float_workspace, plan.float_capacity, float_bytes, false)) != cudaSuccess ||
      (status = plan.reserve(plan.int_workspace, plan.int_capacity, int_bytes, false)) != cudaSuccess ||
      (status = plan.reserve(plan.pinned_int_workspace, plan.pinned_capacity, int_bytes, true)) != cudaSuccess) {
    return status;
  }
  return flashinfer::DecodePlan<128, flashinfer::PosEncodingMode::kNone,
                                Variant, Params>(
      plan.float_workspace, plan.float_capacity, plan.int_workspace,
      plan.pinned_int_workspace, plan.int_capacity, plan.info,
      const_cast<int32_t*>(indptr_host), batch_size, num_q_heads,
      page_size, graph, stream, estimate);
}

template <typename T>
cudaError_t dispatch_plan(NativePlan& plan, const int32_t* indptr_host,
                          uint32_t batch_size, uint32_t num_q_heads,
                          uint32_t num_kv_heads, uint32_t page_size, bool graph,
                          cudaStream_t stream) {
  cudaError_t status = cudaErrorInvalidValue;
  DISPATCH_GQA_GROUP_SIZE(num_q_heads / num_kv_heads, GROUP_SIZE, {
    status = make_plan<T, GROUP_SIZE>(plan, indptr_host, batch_size,
                                      num_q_heads, page_size, graph, stream);
  });
  return status;
}

template <typename T>
cudaError_t run(const NativePlan& plan, const void* query,
                const void* key_cache, const void* value_cache,
                const int32_t* page_indices, const int32_t* page_indptr,
                const int32_t* page_last_len, void* output,
                uint32_t batch_size, uint32_t num_q_heads,
                uint32_t num_kv_heads, uint32_t page_size, cudaStream_t stream) {
  using namespace flashinfer;
  using Params = BatchDecodeParams<T, T, T, int32_t>;
  paged_kv_t<T, int32_t> paged_kv(
      num_kv_heads, page_size, 128, batch_size, QKVLayout::kNHD,
      reinterpret_cast<T*>(const_cast<void*>(key_cache)),
      reinterpret_cast<T*>(const_cast<void*>(value_cache)),
      const_cast<int32_t*>(page_indices), const_cast<int32_t*>(page_indptr),
      const_cast<int32_t*>(page_last_len));
  Params params;
  params.q = reinterpret_cast<T*>(const_cast<void*>(query));
  params.paged_kv = paged_kv;
  params.o = reinterpret_cast<T*>(output);
  params.num_qo_heads = num_q_heads;
  params.padded_batch_size = plan.info.padded_batch_size;
  params.q_stride_n = num_q_heads * 128;
  params.q_stride_h = 128;
  params.window_left = -1;
  params.sm_scale = 1.f / std::sqrt(128.f);
  params.request_indices = GetPtrFromBaseOffset<int32_t>(
      plan.int_workspace, plan.info.request_indices_offset);
  params.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(
      plan.int_workspace, plan.info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<int32_t>(
      plan.int_workspace, plan.info.o_indptr_offset);
  params.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(
      plan.int_workspace, plan.info.kv_chunk_size_ptr_offset);
  if (plan.info.enable_cuda_graph && plan.info.split_kv) {
    params.block_valid_mask = GetPtrFromBaseOffset<bool>(plan.int_workspace,
        plan.info.block_valid_mask_offset);
  }
  T* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan.info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<T>(plan.float_workspace, plan.info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(plan.float_workspace, plan.info.s_offset);
  }
  return BatchDecodeWithPagedKVCacheDispatched<128, PosEncodingMode::kNone,
                                               Variant>(params, tmp_v, tmp_s,
                                                        false, stream);
}
}  // namespace

extern "C" void* sglang_flashinfer_native_plan(
    void* existing, const int32_t* indptr_host, int32_t batch_size, int32_t num_q_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t dtype_code, bool graph,
    void* stream, const char** error) {
  try {
    auto created = existing ? nullptr : std::make_unique<NativePlan>();
    auto* plan = existing ? static_cast<NativePlan*>(existing) : created.get();
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = dtype_code == 0
        ? dispatch_plan<__nv_bfloat16>(*plan, indptr_host, batch_size,
                                      num_q_heads, num_kv_heads, page_size, graph,
                                      cuda_stream)
        : dispatch_plan<__half>(*plan, indptr_host, batch_size,
                                num_q_heads, num_kv_heads, page_size, graph,
                                cuda_stream);
    if (status != cudaSuccess) {
      *error = cudaGetErrorString(status);
      return nullptr;
    }
    return existing ? existing : created.release();
  } catch (const std::exception& e) {
    static thread_local std::string message;
    message = e.what();
    *error = message.c_str();
    return nullptr;
  }
}

extern "C" uint64_t sglang_flashinfer_native_allocations(const void* plan) {
  return static_cast<const NativePlan*>(plan)->allocations;
}

extern "C" void sglang_flashinfer_native_plan_drop(void* plan) {
  delete static_cast<NativePlan*>(plan);
}

extern "C" const char* sglang_flashinfer_native_run(
    const void* plan, const void* query, const void* key_cache,
    const void* value_cache, const int32_t* page_indices,
    const int32_t* page_indptr, const int32_t* page_last_len, void* output,
    int32_t batch_size, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t page_size, int32_t dtype_code, void* stream) {
  const auto& native = *static_cast<const NativePlan*>(plan);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t status = dtype_code == 0
      ? run<__nv_bfloat16>(native, query, key_cache, value_cache,
                           page_indices, page_indptr, page_last_len, output,
                           batch_size, num_q_heads, num_kv_heads, page_size,
                           cuda_stream)
      : run<__half>(native, query, key_cache, value_cache,
                    page_indices, page_indptr, page_last_len, output,
                    batch_size, num_q_heads, num_kv_heads, page_size,
                    cuda_stream);
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}

// Plan once per forward; all decoder layers share the same workspaces.
extern "C" void* sglang_flashinfer_native_prefill_plan(
    void* existing, int32_t* q_host, int32_t* kv_host, int32_t rows,
    int32_t batch, int32_t q_heads, int32_t kv_heads, int32_t page_size,
    void* stream, const char** error) {
  try {
    using namespace flashinfer;
    auto created = existing ? nullptr : std::make_unique<NativePlan>();
    auto* plan = existing ? static_cast<NativePlan*>(existing) : created.get();
    int device, sms;
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess)
      status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) { *error = cudaGetErrorString(status); return nullptr; }
    auto estimate = PrefillSplitQOKVIndptr(q_host, kv_host, rows, batch,
        q_heads, kv_heads, 128, page_size, 2 * sms / kv_heads, false, -1, -1, false);
    const size_t padded = std::get<2>(estimate), tile = std::get<3>(estimate);
    const size_t int_bytes = padded * 13 + (size_t(batch) + rows + 2) * 4 + 256;
    const size_t float_bytes = std::get<0>(estimate)
        ? size_t(q_heads) * padded * tile * 129 * sizeof(float) + 64 : 1;
    if ((status = plan->reserve(plan->float_workspace, plan->float_capacity, float_bytes, false)) != cudaSuccess ||
        (status = plan->reserve(plan->int_workspace, plan->int_capacity, int_bytes, false)) != cudaSuccess ||
        (status = plan->reserve(plan->pinned_int_workspace, plan->pinned_capacity, int_bytes, true)) != cudaSuccess) {
      *error = cudaGetErrorString(status); return nullptr;
    }
    status = PrefillPlan<int32_t>(plan->float_workspace, plan->float_capacity,
        plan->int_workspace, plan->pinned_int_workspace, plan->int_capacity,
        plan->prefill_info, q_host, kv_host, rows, batch, q_heads, kv_heads,
        128, 128, page_size, false, 2, -1, -1, false, 0,
        reinterpret_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) { *error = cudaGetErrorString(status); return nullptr; }
    return existing ? existing : created.release();
  } catch (const std::exception& e) {
    static thread_local std::string message;
    message = e.what(); *error = message.c_str(); return nullptr;
  }
}

namespace {
template <typename T, bool Paged>
cudaError_t run_prefill(const NativePlan& plan, void* q, void* k, void* v,
    int32_t* q_indptr, int32_t* kv_indptr, int32_t* indices, int32_t* last_len,
    void* output, int32_t batch, int32_t q_heads, int32_t kv_heads,
    int32_t page_size, int64_t q_stride, int64_t k_stride, int64_t v_stride,
    cudaStream_t stream) {
  using namespace flashinfer;
  using Params = std::conditional_t<Paged, BatchPrefillPagedParams<T,T,T,int32_t>,
      BatchPrefillRaggedParams<T,T,T,int32_t>>;
  Params params;
  params.q = static_cast<T*>(q);
  params.o = static_cast<T*>(output);
  params.q_indptr = q_indptr;
  params.num_qo_heads = q_heads;
  params.group_size = uint_fastdiv(q_heads / kv_heads);
  params.q_stride_n = q_stride; params.q_stride_h = 128;
  params.window_left = -1; params.sm_scale = 1.f / std::sqrt(128.f);
  if constexpr (Paged) {
    params.paged_kv = paged_kv_t<T,int32_t>(kv_heads, page_size, 128, batch,
        QKVLayout::kNHD, static_cast<T*>(k), static_cast<T*>(v), indices, kv_indptr, last_len);
  } else {
    params.k = static_cast<T*>(k); params.v = static_cast<T*>(v);
    params.kv_indptr = kv_indptr; params.num_kv_heads = kv_heads;
    params.k_stride_n = k_stride; params.k_stride_h = 128;
    params.v_stride_n = v_stride; params.v_stride_h = 128;
  }
  const auto& info = plan.prefill_info;
  params.request_indices = GetPtrFromBaseOffset<int32_t>(plan.int_workspace, info.request_indices_offset);
  params.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(plan.int_workspace, info.qo_tile_indices_offset);
  params.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(plan.int_workspace, info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<int32_t>(plan.int_workspace, info.o_indptr_offset);
  params.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(plan.int_workspace, info.kv_chunk_size_ptr_offset);
  params.padded_batch_size = info.padded_batch_size;
  params.max_total_num_rows = info.total_num_rows;
  T* tmp_v = nullptr; float* tmp_s = nullptr;
  if (info.split_kv) {
    params.merge_indptr = GetPtrFromBaseOffset<int32_t>(plan.int_workspace, info.merge_indptr_offset);
    tmp_v = GetPtrFromBaseOffset<T>(plan.float_workspace, info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(plan.float_workspace, info.s_offset);
  }
  cudaError_t status = cudaErrorInvalidValue;
  DISPATCH_CTA_TILE_Q(info.cta_tile_q, CTA_TILE_Q, {
    if constexpr (Paged) {
      status = BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q,128,128,
          PosEncodingMode::kNone,false,MaskMode::kCausal,Variant>(params,tmp_v,tmp_s,false,stream);
    } else {
      status = BatchPrefillWithRaggedKVCacheDispatched<CTA_TILE_Q,128,128,
          PosEncodingMode::kNone,false,MaskMode::kCausal,Variant>(params,tmp_v,tmp_s,false,stream);
    }
  });
  return status;
}
}

extern "C" const char* sglang_flashinfer_native_prefill_run(
    const void* plan, void* q, void* k, void* v, int32_t* q_indptr,
    int32_t* kv_indptr, int32_t* indices, int32_t* last_len, void* output,
    int32_t batch, int32_t q_heads, int32_t kv_heads, int32_t page_size,
    int64_t q_stride, int64_t k_stride, int64_t v_stride,
    int32_t dtype, bool paged, void* stream) {
  const auto& native = *static_cast<const NativePlan*>(plan);
  cudaError_t status;
#define RUN(T, PAGED) run_prefill<T, PAGED>(native,q,k,v,q_indptr,kv_indptr,indices,last_len,output,batch,q_heads,kv_heads,page_size,q_stride,k_stride,v_stride,reinterpret_cast<cudaStream_t>(stream))
  if (dtype == 0) status = paged ? RUN(__nv_bfloat16,true) : RUN(__nv_bfloat16,false);
  else status = paged ? RUN(__half,true) : RUN(__half,false);
#undef RUN
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}
