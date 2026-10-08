// Upstream device kernels; TensorView host wrappers removed.
#pragma once
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/impl/norm.cuh>
namespace sglang {

struct QKNormParams {
  void* __restrict__ q;
  void* __restrict__ k;  // k is offset by (-num_qo_heads * head_dim) elements
  int64_t q_stride;
  int64_t k_stride;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  float eps;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  uint32_t num_tokens;
};

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kThreadsPerBlock = kWarpsPerBlock * device::kWarpThreads;

// Warp-level kernel for head_dim <= 256
template <int64_t kHeadDim, bool kUsePDL, typename Float>
__global__ void fused_qknorm_warp(const QKNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kHeadDim>;

  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto num_blks = gridDim.x;
  const auto num_workers = num_blks * kWarpsPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto start_worker_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;
  const auto gmem = tile::Memory<Storage>::warp();

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    const auto input_vec = gmem.load(input);
    const auto weight_vec = gmem.load(weight);
    const auto output_vec = norm::apply_norm_warp<kHeadDim>(input_vec, weight_vec, eps);
    gmem.store(input, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// For CTA level, used for head_dim > 256 (512,1024)
template <int64_t kHeadDim, bool kUsePDL, typename Float>
__global__ void fused_qknorm_cta(const QKNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kHeadDim>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<Float, kHeadDim>();
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto idx = blockIdx.x; idx < num_works; idx += gridDim.x) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    const auto input_vec = gmem.load(input);
    const auto weight_vec = gmem.load(weight);
    const auto output_vec = norm::apply_norm_cta<kHeadDim>(input_vec, weight_vec, eps, smem, kNumWarps);
    gmem.store(input, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}


} // namespace sglang
