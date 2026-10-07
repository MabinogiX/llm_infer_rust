#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

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

const char* launch_error() {
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}
}  // namespace

extern "C" const char* sglang_qwen3_native_norm(
    const void* x, const void* weight, void* out, int64_t tokens,
    int64_t heads, int64_t width, int64_t token_stride, int64_t head_stride,
    double eps, int dtype, void* stream) {
  const dim3 grid(tokens * heads);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
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

extern "C" const char* sglang_qwen3_native_qk_norm(
    void* q, void* k, const void* qw, const void* kw,
    int64_t tokens, int64_t qheads, int64_t kheads,
    int64_t width, int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    double eps, int dtype, void* stream) {
  const dim3 grid(tokens * (qheads + kheads));
  auto cuda_stream = static_cast<cudaStream_t>(stream);
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

extern "C" const char* sglang_qwen3_native_rope(
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

extern "C" const char* sglang_qwen3_native_silu_mul(
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
