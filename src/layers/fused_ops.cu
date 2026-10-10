#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <qknorm.cuh>

#ifdef SGLANG_USE_FLASHINFER_NORM
#include <flashinfer/norm.cuh>
#endif

namespace {

template <typename T> __device__ float as_float(T x);
template <> __device__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ float as_float(__half x) { return __half2float(x); }

template <typename T> __device__ T from_float(float x);
template <> __device__ __nv_bfloat16 from_float(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __half from_float(float x) { return __float2half_rn(x); }

__device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ float block_sum(float value) {
  __shared__ float sums[8];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (lane == 0) sums[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? sums[lane] : 0.0f;
  if (warp == 0) value = warp_sum(value);
  __shared__ float total;
  if (threadIdx.x == 0) total = value;
  __syncthreads();
  return total;
}

template <typename T>
__device__ void norm_row(const T* input, const T* weight, T* output,
                         int64_t width, float eps) {
  float sum = 0.0f;
  for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float x = as_float(input[i]);
    sum += x * x;
  }
  const float scale = rsqrtf(block_sum(sum) / width + eps);
  for (int64_t i = threadIdx.x; i < width; i += blockDim.x)
    output[i] = from_float<T>(as_float(input[i]) * scale * as_float(weight[i]));
}

template <typename T>
__global__ void norm_kernel(const T* x, const T* weight, T* out,
                            int64_t tokens, int64_t heads, int64_t width,
                            int64_t token_stride, int64_t head_stride, float eps) {
  const int64_t row = blockIdx.x;
  if (row >= tokens * heads) return;
  norm_row(x + (row / heads) * token_stride + (row % heads) * head_stride,
           weight, out + row * width, width, eps);
}

template <typename T>
__global__ void qk_norm_kernel(
    T* q, T* k, const T* qw, const T* kw,
    int64_t tokens, int64_t qheads, int64_t kheads, int64_t width,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride, float eps) {
  const int64_t row = blockIdx.x;
  const int64_t token = row / (qheads + kheads);
  const int64_t head = row % (qheads + kheads);
  if (token >= tokens) return;
  if (head < qheads) {
    T* row_ptr = q + token * q_token_stride + head * q_head_stride;
    norm_row(row_ptr, qw, row_ptr, width, eps);
  } else {
    const int64_t kh = head - qheads;
    T* row_ptr = k + token * k_token_stride + kh * k_head_stride;
    norm_row(row_ptr, kw, row_ptr, width, eps);
  }
}

template <typename T>
__global__ void add_norm_kernel(T* x, T* residual, const T* weight,
                                int64_t width, int64_t x_stride,
                                int64_t residual_stride, float eps) {
  T* input = x + blockIdx.x * x_stride;
  T* sum_row = residual + blockIdx.x * residual_stride;
  float sum = 0.0f;
  for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float value = as_float(input[i]) + as_float(sum_row[i]);
    sum += value * value;
  }
  const float scale = rsqrtf(block_sum(sum) / width + eps);
  for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float value = as_float(input[i]) + as_float(sum_row[i]);
    sum_row[i] = from_float<T>(value);
    input[i] = from_float<T>(value * scale * as_float(weight[i]));
  }
}

template <typename T>
__global__ void rope_kernel(
    T* q, T* k, const int64_t* positions, const float* cos,
    const float* sin, int64_t tokens, int64_t qheads,
    int64_t kheads, int64_t width, int64_t q_token_stride,
    int64_t q_head_stride, int64_t k_token_stride,
    int64_t k_head_stride, int64_t max_positions) {
  const int64_t row = blockIdx.x;
  const int64_t token = row / (qheads + kheads);
  const int64_t head = row % (qheads + kheads);
  if (token >= tokens) return;
  const int64_t position = positions[token];
  if (position < 0 || position >= max_positions) return;
  const int64_t half = width / 2;
  T* data = head < qheads
      ? q + token * q_token_stride + head * q_head_stride
      : k + token * k_token_stride + (head - qheads) * k_head_stride;
  for (int64_t i = threadIdx.x; i < half; i += blockDim.x) {
    const float first = as_float(data[i]);
    const float second = as_float(data[i + half]);
    const float c = cos[position * half + i];
    const float s = sin[position * half + i];
    data[i] = from_float<T>(first * c - second * s);
    data[i + half] = from_float<T>(second * c + first * s);
  }
}

template <typename T>
__global__ void silu_mul_kernel(const T* gate_up, T* out,
                                int64_t tokens, int64_t width) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * width) return;
  const int64_t token = index / width;
  const int64_t column = index % width;
  const float gate = as_float(gate_up[token * 2 * width + column]);
  const float up = as_float(gate_up[token * 2 * width + width + column]);
  out[index] = from_float<T>((gate / (1.0f + expf(-gate))) * up);
}

// Eight BF16/FP16 values form one 16-byte global-memory transaction.
template <typename T>
__global__ void silu_mul_vector_kernel(const T* gate_up, T* out,
                                       int64_t tokens, int64_t width) {
  constexpr int kValuesPerVector = 8;
  const int64_t vectors_per_token = width / kValuesPerVector;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * vectors_per_token) return;
  const int64_t token = index / vectors_per_token;
  const int64_t vector = index % vectors_per_token;
  const int64_t input_vector = token * 2 * vectors_per_token + vector;
  const uint4 gate_pack = reinterpret_cast<const uint4*>(gate_up)[input_vector];
  const uint4 up_pack = reinterpret_cast<const uint4*>(gate_up)[input_vector + vectors_per_token];
  uint4 result;
  const T* gate_values = reinterpret_cast<const T*>(&gate_pack);
  const T* up_values = reinterpret_cast<const T*>(&up_pack);
  T* output_values = reinterpret_cast<T*>(&result);
#pragma unroll
  for (int i = 0; i < kValuesPerVector; ++i) {
    const float gate = as_float(gate_values[i]);
    const float up = as_float(up_values[i]);
    output_values[i] = from_float<T>((gate / (1.0f + expf(-gate))) * up);
  }
  reinterpret_cast<uint4*>(out)[index] = result;
}

// Launch the vendored upstream device algorithm on the ATen current stream.
template <int64_t D, typename T>
const char* launch_upstream_qk(sglang::QKNormParams params, cudaStream_t stream) {
  constexpr bool warp = D <= 256;
  constexpr int threads = [] {
    if constexpr (warp) return 128;
    else return int(sglang::host::norm::get_cta_threads<T, D>());
  }();
  auto kernel = [] {
    if constexpr (warp) return sglang::fused_qknorm_warp<D, false, T>;
    else return sglang::fused_qknorm_cta<D, false, T>;
  }();
  int occupancy = 1, device = 0, sms = 1;
  auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, threads, 0);
  if (status != cudaSuccess) return cudaGetErrorString(status);
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) return cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return cudaGetErrorString(status);
  const int64_t works = int64_t(params.num_tokens) * (params.num_qo_heads + params.num_kv_heads);
  const int64_t needed = warp ? (works + 3) / 4 : works;
  const dim3 blocks(std::min<int64_t>(sms * occupancy, needed));
  void* args[] = {&params};
  status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel), blocks, dim3(threads), args, 0, stream);
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}

bool aligned(const void* pointer, uintptr_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

const char* launch_error() {
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}
}  // namespace

extern "C" const char* sglang_layers_native_norm(
    const void* x, const void* weight, void* out, int64_t tokens,
    int64_t heads, int64_t width, int64_t token_stride, int64_t head_stride,
    double eps, int dtype, void* stream) {
  const dim3 grid(tokens * heads);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (tokens == 0) return nullptr;
#ifdef SGLANG_USE_FLASHINFER_NORM
  // Official FlashInfer implementation supports a separate input row stride.
  // Its vector loads require aligned row starts; unusual layouts use fallback.
  if (heads == 1 && width > 0 && width % 8 == 0 && token_stride % 8 == 0 &&
      width <= std::numeric_limits<uint32_t>::max() &&
      tokens <= std::numeric_limits<uint32_t>::max() &&
      token_stride > 0 && token_stride <= std::numeric_limits<uint32_t>::max() &&
      aligned(x, 16) && aligned(weight, 16) && aligned(out, 16)) {
    cudaError_t status;
    if (dtype == 0)
      status = flashinfer::norm::RMSNorm(
          const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(x)),
          const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
          static_cast<__nv_bfloat16*>(out), tokens, width, token_stride, width,
          static_cast<float>(eps), false, cuda_stream);
    else
      status = flashinfer::norm::RMSNorm(
          const_cast<__half*>(static_cast<const __half*>(x)),
          const_cast<__half*>(static_cast<const __half*>(weight)),
          static_cast<__half*>(out), tokens, width, token_stride, width,
          static_cast<float>(eps), false, cuda_stream);
    return status == cudaSuccess ? launch_error() : cudaGetErrorString(status);
  }
#endif
  if (dtype == 0)
    norm_kernel<<<grid, 256, 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(out), tokens, heads, width, token_stride,
        head_stride, static_cast<float>(eps));
  else
    norm_kernel<<<grid, 256, 0, cuda_stream>>>(
        static_cast<const __half*>(x), static_cast<const __half*>(weight),
        static_cast<__half*>(out), tokens, heads, width, token_stride,
        head_stride, static_cast<float>(eps));
  return launch_error();
}

extern "C" const char* sglang_layers_native_qk_norm(
    void* q, void* k, const void* qw, const void* kw,
    int64_t tokens, int64_t qheads, int64_t kheads,
    int64_t width, int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    double eps, int dtype, void* stream) {
  const dim3 grid(tokens * (qheads + kheads));
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (tokens == 0) return nullptr;
  if ((width == 64 || width == 128 || width == 256 || width == 512 || width == 1024) &&
      q_head_stride == width && k_head_stride == width &&
      q_token_stride % 8 == 0 && k_token_stride % 8 == 0 &&
      aligned(q, 16) && aligned(k, 16) && aligned(qw, 16) && aligned(kw, 16)) {
    sglang::QKNormParams params{q,
        static_cast<char*>(k) - 2 * qheads * width,
        q_token_stride, k_token_stride, static_cast<uint32_t>(qheads),
        static_cast<uint32_t>(kheads), static_cast<float>(eps), qw, kw,
        static_cast<uint32_t>(tokens)};
#define LAUNCH_QK(D) \
    if (dtype == 0) return launch_upstream_qk<D, __nv_bfloat16>(params, cuda_stream); \
    else return launch_upstream_qk<D, __half>(params, cuda_stream)
    switch (width) {
      case 64: LAUNCH_QK(64); break;
      case 128: LAUNCH_QK(128); break;
      case 256: LAUNCH_QK(256); break;
      case 512: LAUNCH_QK(512); break;
      case 1024: LAUNCH_QK(1024); break;
    }
#undef LAUNCH_QK
    return launch_error();
  }
  if (dtype == 0)
    qk_norm_kernel<<<grid, 256, 0, cuda_stream>>>(
        static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(qw), static_cast<const __nv_bfloat16*>(kw),
        tokens, qheads, kheads, width, q_token_stride, q_head_stride,
        k_token_stride, k_head_stride, static_cast<float>(eps));
  else
    qk_norm_kernel<<<grid, 256, 0, cuda_stream>>>(
        static_cast<__half*>(q), static_cast<__half*>(k),
        static_cast<const __half*>(qw), static_cast<const __half*>(kw),
        tokens, qheads, kheads, width, q_token_stride, q_head_stride,
        k_token_stride, k_head_stride, static_cast<float>(eps));
  return launch_error();
}

extern "C" const char* sglang_layers_native_add_norm(
    void* x, void* residual, const void* weight, int64_t tokens,
    int64_t width, int64_t x_stride, int64_t residual_stride,
    double eps, int dtype, void* stream) {
  if (tokens == 0) return nullptr;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
#ifdef SGLANG_USE_FLASHINFER_NORM
  if (width > 0 && width <= 16384 && width % 8 == 0 &&
      x_stride > 0 && residual_stride > 0 && x_stride % 8 == 0 && residual_stride % 8 == 0 &&
      tokens <= std::numeric_limits<uint32_t>::max() &&
      x_stride <= std::numeric_limits<uint32_t>::max() &&
      residual_stride <= std::numeric_limits<uint32_t>::max() &&
      aligned(x, 16) && aligned(residual, 16) && aligned(weight, 16)) {
    cudaError_t status;
    if (dtype == 0)
      status = flashinfer::norm::FusedAddRMSNorm(
          static_cast<__nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(residual),
          const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(weight)),
          tokens, width, x_stride, residual_stride, static_cast<float>(eps), false, cuda_stream);
    else
      status = flashinfer::norm::FusedAddRMSNorm(
          static_cast<__half*>(x), static_cast<__half*>(residual),
          const_cast<__half*>(static_cast<const __half*>(weight)),
          tokens, width, x_stride, residual_stride, static_cast<float>(eps), false, cuda_stream);
    return status == cudaSuccess ? launch_error() : cudaGetErrorString(status);
  }
#endif
  if (dtype == 0)
    add_norm_kernel<<<tokens, 256, 0, cuda_stream>>>(
        static_cast<__nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(weight), width, x_stride, residual_stride,
        static_cast<float>(eps));
  else
    add_norm_kernel<<<tokens, 256, 0, cuda_stream>>>(
        static_cast<__half*>(x), static_cast<__half*>(residual),
        static_cast<const __half*>(weight), width, x_stride, residual_stride,
        static_cast<float>(eps));
  return launch_error();
}

extern "C" const char* sglang_layers_native_rope(
    void* q, void* k, const int64_t* positions,
    const float* cos, const float* sin, int64_t tokens,
    int64_t qheads, int64_t kheads, int64_t width,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride, int64_t max_positions,
    int dtype, void* stream) {
  const dim3 grid(tokens * (qheads + kheads));
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (dtype == 0)
    rope_kernel<<<grid, 128, 0, cuda_stream>>>(
        static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
        positions, cos, sin, tokens, qheads, kheads, width,
        q_token_stride, q_head_stride, k_token_stride, k_head_stride,
        max_positions);
  else
    rope_kernel<<<grid, 128, 0, cuda_stream>>>(
        static_cast<__half*>(q), static_cast<__half*>(k),
        positions, cos, sin, tokens, qheads, kheads, width,
        q_token_stride, q_head_stride, k_token_stride, k_head_stride,
        max_positions);
  return launch_error();
}

extern "C" const char* sglang_layers_native_silu_mul(
    const void* gate_up, void* out, int64_t tokens, int64_t width,
    int dtype, void* stream) {
  const dim3 grid((tokens * width + 255) / 256);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  const bool vectorized = width % 8 == 0 &&
      reinterpret_cast<uintptr_t>(gate_up) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(out) % alignof(uint4) == 0;
  if (vectorized) {
    const dim3 vector_grid((tokens * (width / 8) + 127) / 128);
    if (dtype == 0)
      silu_mul_vector_kernel<<<vector_grid, 128, 0, cuda_stream>>>(
          static_cast<const __nv_bfloat16*>(gate_up),
          static_cast<__nv_bfloat16*>(out), tokens, width);
    else
      silu_mul_vector_kernel<<<vector_grid, 128, 0, cuda_stream>>>(
          static_cast<const __half*>(gate_up), static_cast<__half*>(out),
          tokens, width);
    return launch_error();
  }
  if (dtype == 0)
    silu_mul_kernel<<<grid, 256, 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(out),
        tokens, width);
  else
    silu_mul_kernel<<<grid, 256, 0, cuda_stream>>>(
        static_cast<const __half*>(gate_up), static_cast<__half*>(out),
        tokens, width);
  return launch_error();
}
