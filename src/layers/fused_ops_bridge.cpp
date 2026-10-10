#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

extern "C" const char* sglang_layers_native_norm(
    const void*, const void*, void*, int64_t, int64_t, int64_t, int64_t,
    int64_t, double, int, void*);
extern "C" const char* sglang_layers_native_add_norm(
    void*, void*, const void*, int64_t, int64_t, int64_t, int64_t, double, int, void*);
extern "C" const char* sglang_layers_native_qk_norm(
    void*, void*, const void*, const void*, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, double, int, void*);
extern "C" const char* sglang_layers_native_rope(
    void*, void*, const int64_t*, const float*, const float*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int, void*);
extern "C" const char* sglang_layers_native_silu_mul(
    const void*, void*, int64_t, int64_t, int, void*);

int dtype_code(const at::Tensor& x) {
  if (!x.is_cuda()) throw std::runtime_error("expected CUDA tensor");
  if (x.scalar_type() == at::kBFloat16) return 0;
  if (x.scalar_type() == at::kHalf) return 1;
  throw std::runtime_error("expected bfloat16 or float16 tensor");
}

void check(const char* error) {
  if (error) throw std::runtime_error(error);
}

void check_weight(const at::Tensor& x, const at::Tensor& weight, int64_t dim) {
  if (!weight.is_cuda() || weight.get_device() != x.get_device() ||
      weight.scalar_type() != x.scalar_type() || weight.dim() != 1 ||
      weight.size(0) != dim || !weight.is_contiguous()) {
    throw std::runtime_error("invalid shared layer norm weight");
  }
}

void check_qk(const at::Tensor& q, const at::Tensor& k) {
  if (q.dim() != 3 || k.dim() != 3 || q.size(0) != k.size(0) ||
      q.size(2) != k.size(2) || q.get_device() != k.get_device() ||
      q.scalar_type() != k.scalar_type() || q.stride(2) != 1 ||
      k.stride(2) != 1) {
    throw std::runtime_error("invalid shared layer query/key shapes or strides");
  }
}
}  // namespace

extern "C" const char* sglang_layers_error() { return last_error.c_str(); }

extern "C" at::Tensor* sglang_layers_rms_norm(
    const at::Tensor* x, const at::Tensor* weight, double eps) {
  try {
    const int dtype = dtype_code(*x);
    if (x->dim() != 2 || x->stride(1) != 1)
      throw std::runtime_error("invalid shared layer RMSNorm input");
    check_weight(*x, *weight, x->size(1));
    auto out = std::make_unique<at::Tensor>(at::empty_like(*x, at::MemoryFormat::Contiguous));
    auto stream = c10::cuda::getCurrentCUDAStream(x->get_device()).stream();
    check(sglang_layers_native_norm(x->data_ptr(), weight->data_ptr(),
                                   out->data_ptr(), x->size(0), 1, x->size(1),
                                   x->stride(0), 0, eps, dtype, stream));
    return out.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" bool sglang_layers_add_rms_norm_inplace(
    const at::Tensor* x, const at::Tensor* residual, const at::Tensor* weight, double eps) {
  try {
    const int dtype = dtype_code(*x);
    if (x->dim() != 2 || x->stride(1) != 1 ||
        !residual->is_cuda() || residual->get_device() != x->get_device() ||
        residual->scalar_type() != x->scalar_type() ||
        residual->sizes() != x->sizes() || residual->stride(1) != 1 ||
        x->data_ptr() == residual->data_ptr())
      throw std::runtime_error("invalid shared layer fused add-RMSNorm inputs");
    check_weight(*x, *weight, x->size(1));
    auto stream = c10::cuda::getCurrentCUDAStream(x->get_device()).stream();
    check(sglang_layers_native_add_norm(
        x->data_ptr(), residual->data_ptr(), weight->data_ptr(),
        x->size(0), x->size(1), x->stride(0), residual->stride(0), eps, dtype, stream));
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" bool sglang_layers_qk_norm_inplace(
    const at::Tensor* q, const at::Tensor* k, const at::Tensor* qw,
    const at::Tensor* kw, double eps) {
  try {
    const int dtype = dtype_code(*q);
    check_qk(*q, *k);
    check_weight(*q, *qw, q->size(2));
    check_weight(*k, *kw, k->size(2));
    auto stream = c10::cuda::getCurrentCUDAStream(q->get_device()).stream();
    check(sglang_layers_native_qk_norm(
        q->data_ptr(), k->data_ptr(), qw->data_ptr(), kw->data_ptr(),
        q->size(0), q->size(1), k->size(1), q->size(2),
        q->stride(0), q->stride(1), k->stride(0), k->stride(1),
        eps, dtype, stream));
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" bool sglang_layers_rope_inplace(
    const at::Tensor* q, const at::Tensor* k, const at::Tensor* positions,
    const at::Tensor* cos, const at::Tensor* sin) {
  try {
    const int dtype = dtype_code(*q);
    check_qk(*q, *k);
    if (q->size(2) % 2 ||
        !positions->is_cuda() || positions->get_device() != q->get_device() ||
        positions->scalar_type() != at::kLong || positions->dim() != 1 ||
        positions->size(0) != q->size(0) || !positions->is_contiguous() ||
        !cos->is_cuda() || !sin->is_cuda() ||
        cos->get_device() != q->get_device() ||
        sin->get_device() != q->get_device() ||
        cos->scalar_type() != at::kFloat ||
        sin->scalar_type() != at::kFloat ||
        cos->sizes() != sin->sizes() || cos->dim() != 2 ||
        cos->size(1) != q->size(2) / 2 ||
        !cos->is_contiguous() || !sin->is_contiguous()) {
      throw std::runtime_error("invalid shared layer RoPE inputs");
    }
    auto stream = c10::cuda::getCurrentCUDAStream(q->get_device()).stream();
    check(sglang_layers_native_rope(
        q->data_ptr(), k->data_ptr(), positions->data_ptr<int64_t>(),
        cos->data_ptr<float>(), sin->data_ptr<float>(),
        q->size(0), q->size(1), k->size(1), q->size(2),
        q->stride(0), q->stride(1), k->stride(0), k->stride(1),
        cos->size(0), dtype, stream));
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" at::Tensor* sglang_layers_silu_and_mul(const at::Tensor* gate_up) {
  try {
    const int dtype = dtype_code(*gate_up);
    if (gate_up->dim() != 2 || gate_up->size(1) % 2 ||
        !gate_up->is_contiguous()) {
      throw std::runtime_error("invalid shared layer gate/up tensor");
    }
    const int64_t width = gate_up->size(1) / 2;
    auto out = std::make_unique<at::Tensor>(at::empty(
        {gate_up->size(0), width}, gate_up->options()));
    auto stream = c10::cuda::getCurrentCUDAStream(gate_up->get_device()).stream();
    check(sglang_layers_native_silu_mul(gate_up->data_ptr(), out->data_ptr(),
                                       gate_up->size(0), width, dtype, stream));
    return out.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}
