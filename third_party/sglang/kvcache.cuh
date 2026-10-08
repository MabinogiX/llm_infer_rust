// Upstream device kernels; TensorView host wrappers removed.
#pragma once
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/vec.cuh>
#include <cassert>
namespace sglang {

struct StoreKVCacheParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  int64_t stride_k_bytes;
  int64_t stride_v_bytes;
  // Independent slot strides: head_dim != v_head_dim gives K and V different row widths.
  int64_t stride_k_cache_bytes;
  int64_t stride_v_cache_bytes;
  int64_t stride_indices;
  uint32_t batch_size;
  int64_t size_limit;
  int64_t reserved_skip_index;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

/**
 * \brief How a warp vectorizes one row of kElementBytes: the widest aligned
 * vector type it can use, and how many full loop iterations that takes.
 * Shared by the interleaved and single-row copies so the two cannot drift.
 * kElementBytes == 0 is a valid (empty) plan, so a zero-width tail can be
 * queried before being branched away.
 */
template <int64_t kElementBytes>
struct RowVecPlan {
  static constexpr int64_t kAlignment = (kElementBytes % (16 * device::kWarpThreads) == 0) ? 16
                                        : kElementBytes % (8 * device::kWarpThreads) == 0  ? 8
                                        : kElementBytes % (4 * device::kWarpThreads) == 0  ? 4
                                        : kElementBytes % 4 == 0                           ? 4
                                                                                           : 0;

  static_assert(kAlignment > 0, "Element size must be multiple of 4 bytes");

  using vec_t = device::AlignedStorage<uint32_t, kAlignment / 4>;
  static constexpr int64_t kLoopBytes = sizeof(vec_t) * device::kWarpThreads;
  static constexpr int64_t kLoopCount = kElementBytes / kLoopBytes;
  static constexpr int64_t kElementCount = kElementBytes / sizeof(vec_t);
  static constexpr bool kHasEpilogue = kLoopCount * kLoopBytes < kElementBytes;
};

/**
 * \brief Use a single warp to copy key and value data from source to destination.
 * Each thread in the warp copies a portion of the data in a coalesced manner.
 * Both loads are issued before either store: the two rows live in different
 * tensors, and the params' __restrict__ does not survive into the kernel body,
 * so the compiler cannot prove k_dst and v_src disjoint and will not sink the
 * V load past the K store on its own.
 * \tparam kElementBytes The size of each key/value element in bytes.
 * \param k_src Pointer to the source key data.
 * \param v_src Pointer to the source value data.
 * \param k_dst Pointer to the destination key data.
 * \param v_dst Pointer to the destination value data.
 */
template <int64_t kElementBytes>
SGL_DEVICE void copy_kv_warp(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  using plan_t = RowVecPlan<kElementBytes>;
  using vec_t = typename plan_t::vec_t;
  constexpr auto kLoopCount = plan_t::kLoopCount;

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto k = gmem.load(k_src, i);
    const auto v = gmem.load(v_src, i);
    gmem.store(k_dst, k, i);
    gmem.store(v_dst, v, i);
  }

  // handle the epilogue if any
  if constexpr (plan_t::kHasEpilogue) {
    if (gmem.in_bound(plan_t::kElementCount, kLoopCount)) {
      const auto k = gmem.load(k_src, kLoopCount);
      const auto v = gmem.load(v_src, kLoopCount);
      gmem.store(k_dst, k, kLoopCount);
      gmem.store(v_dst, v, kLoopCount);
    }
  }
}

/**
 * \brief Use a single warp to copy one row from source to destination.
 * Serves the width by which asymmetric K/V rows differ, which has no counterpart
 * row to interleave with.
 * \tparam kElementBytes The size of the row in bytes.
 * \param src Pointer to the source data.
 * \param dst Pointer to the destination data.
 */
template <int64_t kElementBytes>
SGL_DEVICE void copy_row_warp(const void* __restrict__ src, void* __restrict__ dst) {
  using namespace device;
  using plan_t = RowVecPlan<kElementBytes>;
  using vec_t = typename plan_t::vec_t;
  constexpr auto kLoopCount = plan_t::kLoopCount;

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    gmem.store(dst, gmem.load(src, i), i);
  }

  // handle the epilogue if any
  if constexpr (plan_t::kHasEpilogue) {
    if (gmem.in_bound(plan_t::kElementCount, kLoopCount)) {
      gmem.store(dst, gmem.load(src, kLoopCount), kLoopCount);
    }
  }
}

/**
 * \brief Copy a K row of kKBytes and a V row of kVBytes with one warp.
 * The overlapping prefix goes through the interleaved copy; only the width by
 * which the rows differ is left as a serial tail. Equal widths degenerate to a
 * single interleaved copy with no tail.
 */
template <int64_t kKBytes, int64_t kVBytes>
SGL_DEVICE void copy_kv_rows_warp(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  constexpr auto kCommon = kKBytes < kVBytes ? kKBytes : kVBytes;
  constexpr auto kTail = (kKBytes < kVBytes ? kVBytes : kKBytes) - kCommon;

  // The interleaved copy indexes BOTH rows with kCommon's vector width, so that
  // width must divide each row's split offset -- the narrower row's alignment
  // does not imply the wider one's (e.g. 512 picks 16B, but 516 is not 16B
  // aligned). The tail's own width must likewise divide its kCommon start.
  // Whatever these gates admit is alignment-safe for the strides too, since a
  // stride is a whole multiple of its split size.
  constexpr auto kTailOrCommon = kTail == 0 ? kCommon : kTail;
  constexpr auto kCommonAlign = RowVecPlan<kCommon>::kAlignment;
  constexpr auto kTailAlign = RowVecPlan<kTailOrCommon>::kAlignment;
  constexpr bool kCanInterleave =
      kKBytes % kCommonAlign == 0 && kVBytes % kCommonAlign == 0 && kCommon % kTailAlign == 0;

  if constexpr (kCanInterleave) {
    copy_kv_warp<kCommon>(k_src, v_src, k_dst, v_dst);
    if constexpr (kTail > 0) {
      if constexpr (kKBytes > kVBytes) {
        copy_row_warp<kTail>(pointer::offset(k_src, kCommon), pointer::offset(k_dst, kCommon));
      } else {
        copy_row_warp<kTail>(pointer::offset(v_src, kCommon), pointer::offset(v_dst, kCommon));
      }
    }
  } else {
    copy_row_warp<kKBytes>(k_src, k_dst);
    copy_row_warp<kVBytes>(v_src, v_dst);
  }
}

/**
 * \brief Kernel to store key-value pairs into the KV cache.
 * Each element is split into multiple parts to allow parallel memory copy.
 * \tparam kKElementBytes The size of each key element in bytes.
 * \tparam kVElementBytes The size of each value element in bytes. Differs from
 *         kKElementBytes for asymmetric KV (head_dim != v_head_dim).
 * \tparam kSplit The number of warps that handle each element.
 * \tparam kUsePDL Whether to use PDL feature.
 * \tparam T The data type of the indices (`int32_t` or `int64_t`).
 */
template <int64_t kKElementBytes, int64_t kVElementBytes, int kSplit, bool kUsePDL, typename T>
__global__ void store_kvcache(const __grid_constant__ StoreKVCacheParams params) {
  using namespace device;
  constexpr auto kKSplitSize = kKElementBytes / kSplit;
  constexpr auto kVSplitSize = kVElementBytes / kSplit;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t item_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;
  const auto& [
    k_input, v_input, k_cache, v_cache, indices, // ptr
    stride_k, stride_v, stride_k_cache, stride_v_cache, stride_indices, batch_size, // size
    size_limit, reserved_skip_index // bounds and reserved sink
  ] = params;
  if (item_id >= batch_size) return;

  const auto index_ptr = static_cast<const T*>(indices) + item_id * stride_indices;
  PDLWaitPrimary<kUsePDL>();

  const auto index = *index_ptr;
  // A stale/OOB slot id would cause an illegal memory access in the store below;
  // fail fast at the culprit instead. always-on (kvcache JIT compiles without NDEBUG).
  assert(index >= 0 && index < size_limit);
  const auto k_src = pointer::offset(k_input, item_id * stride_k, split_id * kKSplitSize);
  const auto v_src = pointer::offset(v_input, item_id * stride_v, split_id * kVSplitSize);
  const auto k_dst = pointer::offset(k_cache, index * stride_k_cache, split_id * kKSplitSize);
  const auto v_dst = pointer::offset(v_cache, index * stride_v_cache, split_id * kVSplitSize);

  if (index != reserved_skip_index) {
    copy_kv_rows_warp<kKSplitSize, kVSplitSize>(k_src, v_src, k_dst, v_dst);
  }
  PDLTriggerSecondary<kUsePDL>();
}


} // namespace sglang
