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
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

namespace {
using Variant = flashinfer::DefaultAttention<false, false, false, false>;

struct NativePlan {
  flashinfer::DecodePlanInfo info;
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
                      uint32_t page_size, cudaStream_t stream) {
  using Params = flashinfer::BatchDecodeParams<T, T, T, int32_t>;
  auto estimate = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
      GroupSize, 128, flashinfer::PosEncodingMode::kNone, Variant, Params>;
  bool split_kv = false;
  uint32_t max_grid_size = 0, max_pages = 0, new_batch_size = 0, grid_y = 0;
  auto status = estimate(split_kv, max_grid_size, max_pages, new_batch_size,
                         grid_y, batch_size, const_cast<int32_t*>(indptr_host),
                         num_q_heads, page_size, false, stream);
  if (status != cudaSuccess) return status;

  // FlashInfer's DecodePlan uses four aligned integer arrays and, when
  // split_kv is selected, temporary value and score arrays.
  const size_t padded = std::max<size_t>(new_batch_size, batch_size);
  const size_t int_bytes = padded * 16 + 128;
  const size_t float_bytes = split_kv
      ? size_t(num_q_heads) * padded * (128 * sizeof(float) + sizeof(float)) + 64
      : 1;
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
      page_size, false, stream, estimate);
}

template <typename T>
cudaError_t dispatch_plan(NativePlan& plan, const int32_t* indptr_host,
                          uint32_t batch_size, uint32_t num_q_heads,
                          uint32_t num_kv_heads, uint32_t page_size,
                          cudaStream_t stream) {
  cudaError_t status = cudaErrorInvalidValue;
  DISPATCH_GQA_GROUP_SIZE(num_q_heads / num_kv_heads, GROUP_SIZE, {
    status = make_plan<T, GROUP_SIZE>(plan, indptr_host, batch_size,
                                      num_q_heads, page_size, stream);
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
    int32_t num_kv_heads, int32_t page_size, int32_t dtype_code,
    void* stream, const char** error) {
  try {
    auto created = existing ? nullptr : std::make_unique<NativePlan>();
    auto* plan = existing ? static_cast<NativePlan*>(existing) : created.get();
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = dtype_code == 0
        ? dispatch_plan<__nv_bfloat16>(*plan, indptr_host, batch_size,
                                      num_q_heads, num_kv_heads, page_size,
                                      cuda_stream)
        : dispatch_plan<__half>(*plan, indptr_host, batch_size,
                                num_q_heads, num_kv_heads, page_size,
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
