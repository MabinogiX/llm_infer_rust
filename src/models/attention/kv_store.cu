#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <kvcache.cuh>

namespace {
template <typename Index, bool Vectorized>
__global__ void store_kv_kernel(
    const uint16_t* k, const uint16_t* v, uint16_t* kc, uint16_t* vc,
    const Index* locations, int64_t heads, int64_t width, int64_t capacity,
    int64_t kt, int64_t kh, int64_t vt, int64_t vh, int64_t reserved) {
  const int64_t token = blockIdx.x;
  const int64_t slot = locations[token];
  assert(slot >= 0 && slot < capacity);
  if (slot == reserved) return;
  constexpr int values = Vectorized ? 8 : 1;
  const int64_t units_per_head = width / values;
  for (int64_t unit = threadIdx.x; unit < heads * units_per_head; unit += blockDim.x) {
    const int64_t head = unit / units_per_head;
    const int64_t column = (unit % units_per_head) * values;
    const auto* kp = k + token * kt + head * kh + column;
    const auto* vp = v + token * vt + head * vh + column;
    auto* ko = kc + (slot * heads + head) * width + column;
    auto* vo = vc + (slot * heads + head) * width + column;
    if constexpr (Vectorized) {
      *reinterpret_cast<uint4*>(ko) = *reinterpret_cast<const uint4*>(kp);
      *reinterpret_cast<uint4*>(vo) = *reinterpret_cast<const uint4*>(vp);
    } else {
      *ko = *kp;
      *vo = *vp;
    }
  }
}

template <typename Index>
void launch(const void* k, const void* v, void* kc, void* vc, const void* locations,
            int64_t tokens, int64_t heads, int64_t width, int64_t capacity,
            int64_t kt, int64_t kh, int64_t vt, int64_t vh, int64_t reserved, bool vectorized,
            cudaStream_t stream) {
  if (vectorized)
    store_kv_kernel<Index, true><<<tokens, 128, 0, stream>>>(
        static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v),
        static_cast<uint16_t*>(kc), static_cast<uint16_t*>(vc),
        static_cast<const Index*>(locations), heads, width, capacity, kt, kh, vt, vh, reserved);
  else
    store_kv_kernel<Index, false><<<tokens, 128, 0, stream>>>(
        static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v),
        static_cast<uint16_t*>(kc), static_cast<uint16_t*>(vc),
        static_cast<const Index*>(locations), heads, width, capacity, kt, kh, vt, vh, reserved);
}
const char* launch_error() {
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}

template <int64_t Bytes, typename Index>
void launch_upstream(const sglang::StoreKVCacheParams& params, cudaStream_t stream) {
  constexpr int split = Bytes % 2048 == 0 ? 4 : Bytes % 1024 == 0 ? 2 : 1;
  const auto blocks = (uint64_t(params.batch_size) * split + 3) / 4;
  sglang::store_kvcache<Bytes, Bytes, split, false, Index><<<blocks, 128, 0, stream>>>(params);
}

}  // namespace

extern "C" const char* sglang_native_store_kv(
    const void* k, const void* v, void* kc, void* vc, const void* locations,
    int64_t tokens, int64_t heads, int64_t width, int64_t capacity,
    int64_t kt, int64_t kh, int64_t vt, int64_t vh, int64_t reserved, bool index64, void* stream) {
  if (tokens == 0) return nullptr;
  // Upstream rows are contiguous across head/dim, with arbitrary token stride.
  // Instantiate the common row widths; retain generic handling for other layouts.
  const int64_t bytes = heads * width * 2;
  if (kh == width && vh == width && kt * 2 % 16 == 0 && vt * 2 % 16 == 0 &&
      reinterpret_cast<uintptr_t>(k) % 16 == 0 && reinterpret_cast<uintptr_t>(v) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(kc) % 16 == 0 && reinterpret_cast<uintptr_t>(vc) % 16 == 0) {
    const sglang::StoreKVCacheParams params{k, v, kc, vc, locations,
        kt * 2, vt * 2, bytes, bytes, 1, static_cast<uint32_t>(tokens), capacity, reserved};
#define STORE(B) case B: \
    if (index64) launch_upstream<B, int64_t>(params, static_cast<cudaStream_t>(stream)); \
    else launch_upstream<B, int32_t>(params, static_cast<cudaStream_t>(stream)); \
    return launch_error()
    switch (bytes) {
      STORE(128); STORE(256); STORE(512); STORE(1024); STORE(2048); STORE(4096); STORE(8192);
    }
#undef STORE
  }
  const bool vectorized = width % 8 == 0 && kt % 8 == 0 && kh % 8 == 0 &&
      vt % 8 == 0 && vh % 8 == 0 &&
      reinterpret_cast<uintptr_t>(k) % 16 == 0 && reinterpret_cast<uintptr_t>(v) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(kc) % 16 == 0 && reinterpret_cast<uintptr_t>(vc) % 16 == 0;
  if (index64)
    launch<int64_t>(k, v, kc, vc, locations, tokens, heads, width, capacity, kt, kh, vt, vh, reserved,
                     vectorized, static_cast<cudaStream_t>(stream));
  else
    launch<int32_t>(k, v, kc, vc, locations, tokens, heads, width, capacity, kt, kh, vt, vh, reserved,
                     vectorized, static_cast<cudaStream_t>(stream));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}
