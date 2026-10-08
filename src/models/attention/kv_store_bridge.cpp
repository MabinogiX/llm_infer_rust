#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <stdexcept>

namespace { thread_local std::string last_error; }
extern "C" const char* sglang_native_store_kv(
    const void*, const void*, void*, void*, const void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, void*);
extern "C" const char* sglang_store_kv_error() { return last_error.c_str(); }
extern "C" bool sglang_store_kv(const at::Tensor* k, const at::Tensor* v,
    const at::Tensor* kc, const at::Tensor* vc, const at::Tensor* loc, int64_t reserved_skip_index) {
  try {
    if (!k->is_cuda()) throw std::runtime_error("KV store requires CUDA tensors");
    for (const auto* tensor : {v, kc, vc})
      if (!tensor->is_cuda() || tensor->device() != k->device() ||
          tensor->scalar_type() != k->scalar_type())
        throw std::runtime_error("KV store device/dtype mismatch");
    if ((k->scalar_type() != at::kBFloat16 && k->scalar_type() != at::kHalf) ||
        k->dim() != 3 || v->sizes() != k->sizes() || kc->dim() != 4 || vc->sizes() != kc->sizes() ||
        k->stride(2) != 1 || v->stride(2) != 1 || !kc->is_contiguous() || !vc->is_contiguous() ||
        kc->size(2) != k->size(1) || kc->size(3) != k->size(2) ||
        k->size(1) <= 0 || k->size(2) <= 0)
      throw std::runtime_error("invalid KV store shape/stride");
    if (!loc->is_cuda() || loc->device() != k->device() || loc->dim() != 1 ||
        loc->size(0) != k->size(0) || !loc->is_contiguous() ||
        (loc->scalar_type() != at::kInt && loc->scalar_type() != at::kLong))
      throw std::runtime_error("invalid KV store locations");
    c10::cuda::CUDAGuard guard(k->device());
    const auto* error = sglang_native_store_kv(k->data_ptr(), v->data_ptr(), kc->data_ptr(),
        vc->data_ptr(), loc->data_ptr(), k->size(0), k->size(1), k->size(2),
        kc->size(0)*kc->size(1), k->stride(0), k->stride(1), v->stride(0), v->stride(1),
        reserved_skip_index, loc->scalar_type() == at::kLong, c10::cuda::getCurrentCUDAStream(k->get_device()).stream());
    if (error) throw std::runtime_error(error);
    return true;
  } catch (const std::exception& error) { last_error = error.what(); return false; }
}
