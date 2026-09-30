#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

template <typename T>
const char* launch_decode(
    const void* query, const void* key_cache, const void* value_cache,
    const int32_t* page_indices, const int32_t* page_indptr,
    const int32_t* page_last_len, const int32_t* request_indices,
    const int32_t* kv_tile_indices, const int32_t* kv_chunk_size,
    void* output, int32_t batch_size,
    int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, cudaStream_t stream) {
  using namespace flashinfer;
  using Params = BatchDecodeParams<T, T, T, int32_t>;
  using Variant = DefaultAttention<false, false, false, false>;
  if (head_dim != 128) {
    return "native FlashInfer decode currently requires head_dim=128";
  }
  if (num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) {
    return "invalid FlashInfer attention head counts";
  }
  paged_kv_t<T, int32_t> paged_kv(
      num_kv_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
      reinterpret_cast<T*>(const_cast<void*>(key_cache)),
      reinterpret_cast<T*>(const_cast<void*>(value_cache)),
      const_cast<int32_t*>(page_indices), const_cast<int32_t*>(page_indptr),
      const_cast<int32_t*>(page_last_len));
  Params params;
  params.q = reinterpret_cast<T*>(const_cast<void*>(query));
  params.paged_kv = paged_kv;
  params.o = reinterpret_cast<T*>(output);
  params.num_qo_heads = num_q_heads;
  params.padded_batch_size = batch_size;
  params.q_stride_n = num_q_heads * head_dim;
  params.q_stride_h = head_dim;
  params.window_left = -1;
  params.sm_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
  params.request_indices = const_cast<int32_t*>(request_indices);
  params.kv_tile_indices = const_cast<int32_t*>(kv_tile_indices);
  params.kv_chunk_size_ptr = const_cast<int32_t*>(kv_chunk_size);
  // One CTA per request and KV head. No host-side plan or Python runtime.
  cudaError_t status = BatchDecodeWithPagedKVCacheDispatched<
      128, PosEncodingMode::kNone, Variant>(params, nullptr, nullptr, false, stream);
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}

extern "C" const char* sglang_flashinfer_launch(
    const void* query, const void* key_cache, const void* value_cache,
    const int32_t* page_indices, const int32_t* page_indptr,
    const int32_t* page_last_len, const int32_t* request_indices,
    const int32_t* kv_tile_indices, const int32_t* kv_chunk_size,
    void* output, int32_t batch_size,
    int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t dtype_code, void* stream) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (dtype_code == 0) {
    return launch_decode<__nv_bfloat16>(
        query, key_cache, value_cache, page_indices, page_indptr,
        page_last_len, request_indices, kv_tile_indices, kv_chunk_size,
        output, batch_size, num_q_heads, num_kv_heads,
        head_dim, page_size, cuda_stream);
  }
  if (dtype_code == 1) {
    return launch_decode<__half>(
        query, key_cache, value_cache, page_indices, page_indptr,
        page_last_len, request_indices, kv_tile_indices, kv_chunk_size,
        output, batch_size, num_q_heads, num_kv_heads,
        head_dim, page_size, cuda_stream);
  }
  return "native FlashInfer decode requires BF16 or FP16";
}
