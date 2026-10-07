#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

struct FlashInferPlan {
  at::Tensor q_indptr;
  int64_t total_queries = 0;
  bool prefill = false;
  bool paged = false;
  at::Tensor indices;
  at::Tensor indptr;
  at::Tensor last_page_len;
  void* native_plan = nullptr;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
  int64_t page_size;
  int64_t dtype_code;
  int device_index;
};

extern "C" void* sglang_flashinfer_native_plan(
    void* existing, const int32_t* indptr_host, int32_t batch_size, int32_t num_q_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t dtype_code,
    void* stream, const char** error);
extern "C" void sglang_flashinfer_native_plan_drop(void* plan);
extern "C" uint64_t sglang_flashinfer_native_allocations(const void* plan);
extern "C" const char* sglang_flashinfer_native_run(
    const void* plan, const void* query, const void* key_cache,
    const void* value_cache, const int32_t* page_indices,
    const int32_t* page_indptr, const int32_t* page_last_len, void* output,
    int32_t batch_size, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t page_size, int32_t dtype_code, void* stream);
}  // namespace

extern "C" const char* sglang_flashinfer_error() { return last_error.c_str(); }

extern "C" void* sglang_flashinfer_prepare(
    void* existing, const at::Tensor* block_table, const at::Tensor* sequence_lengths,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t page_size, int64_t dtype_code) {
  try {
    if (!block_table->is_cuda() || !sequence_lengths->is_cuda()) {
      throw std::runtime_error("FlashInfer metadata must be CUDA tensors");
    }
    c10::cuda::CUDAGuard guard(block_table->device());
    if (sequence_lengths->get_device() != block_table->get_device())
      throw std::runtime_error("FlashInfer metadata devices do not match");
    if (block_table->dim() != 2 || sequence_lengths->dim() != 1 ||
        block_table->size(0) != sequence_lengths->size(0)) {
      throw std::runtime_error("invalid FlashInfer block table or lengths");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads ||
        head_dim != 128 || page_size <= 0 || (dtype_code != 0 && dtype_code != 1)) {
      throw std::runtime_error("unsupported FlashInfer attention configuration");
    }
    auto created = existing ? nullptr : std::make_unique<FlashInferPlan>();
    auto* plan = existing ? static_cast<FlashInferPlan*>(existing) : created.get();
    if (existing && (plan->device_index != block_table->get_device() ||
        plan->num_q_heads != num_q_heads || plan->num_kv_heads != num_kv_heads ||
        plan->head_dim != head_dim || plan->page_size != page_size ||
        plan->dtype_code != dtype_code))
      throw std::runtime_error("FlashInfer workspace geometry changed");
    auto pages = at::floor_divide(sequence_lengths->to(at::kInt) + page_size - 1,
                                  page_size);
    plan->indptr = at::cat({at::zeros({1}, pages.options()), pages.cumsum(0, at::kInt)});
    auto columns = at::arange(block_table->size(1), pages.options());
    plan->indices = block_table->masked_select(columns.unsqueeze(0) < pages.unsqueeze(1))
                        .to(at::kInt)
                        .contiguous();
    plan->last_page_len = ((sequence_lengths->to(at::kInt) - 1) % page_size + 1).contiguous();
    auto indptr_host = plan->indptr.to(at::kCPU).contiguous();
    // This blocking read also drains earlier work on the same stream before
    // DecodePlan overwrites its reusable pinned and device workspaces.
    const char* native_error = nullptr;
    auto stream = c10::cuda::getCurrentCUDAStream(block_table->get_device()).stream();
    auto* native_plan = sglang_flashinfer_native_plan(
        plan->native_plan, indptr_host.data_ptr<int32_t>(), block_table->size(0), num_q_heads,
        num_kv_heads, page_size, dtype_code, stream, &native_error);
    if (!native_plan) {
      throw std::runtime_error(native_error ? native_error : "FlashInfer plan failed");
    }
    plan->native_plan = native_plan;
    plan->num_q_heads = num_q_heads;
    plan->num_kv_heads = num_kv_heads;
    plan->head_dim = head_dim;
    plan->page_size = page_size;
    plan->dtype_code = dtype_code;
    plan->device_index = block_table->get_device();
    return existing ? existing : created.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" uint64_t sglang_flashinfer_workspace_allocations(const void* prepared) {
  return sglang_flashinfer_native_allocations(
      static_cast<const FlashInferPlan*>(prepared)->native_plan);
}

extern "C" void sglang_flashinfer_plan_drop(void* plan) {
  auto* prepared = static_cast<FlashInferPlan*>(plan);
  if (prepared) {
    c10::cuda::CUDAGuard guard(prepared->device_index);
    sglang_flashinfer_native_plan_drop(prepared->native_plan);
  }
  delete prepared;
}

extern "C" at::Tensor* sglang_flashinfer_decode(
    const void* prepared, const at::Tensor* query, const at::Tensor* key_cache,
    const at::Tensor* value_cache) {
  try {
    if (!prepared) {
      throw std::runtime_error("FlashInfer decode was not prepared");
    }
    const auto& plan = *static_cast<const FlashInferPlan*>(prepared);
    if (!query->is_cuda() || !key_cache->is_cuda() || !value_cache->is_cuda() ||
        !query->is_contiguous() || !key_cache->is_contiguous() ||
        !value_cache->is_contiguous()) {
      throw std::runtime_error("FlashInfer requires contiguous CUDA tensors");
    }
    const auto batch_size = query->size(0);
    const auto num_q_heads = query->size(1);
    const auto head_dim = query->size(2);
    const auto num_kv_heads = key_cache->size(2);
    const auto page_size = key_cache->size(1);
    const int32_t dtype_code = query->scalar_type() == at::kBFloat16 ? 0 : 1;
    if (num_q_heads != plan.num_q_heads || num_kv_heads != plan.num_kv_heads ||
        head_dim != plan.head_dim || page_size != plan.page_size ||
        dtype_code != plan.dtype_code || batch_size != plan.last_page_len.size(0)) {
      throw std::runtime_error("FlashInfer decode tensors do not match the prepared batch");
    }
    auto output = at::empty_like(*query);
    auto stream = c10::cuda::getCurrentCUDAStream(query->get_device()).stream();
    const char* error = sglang_flashinfer_native_run(
        plan.native_plan, query->data_ptr(), key_cache->data_ptr(), value_cache->data_ptr(),
        plan.indices.data_ptr<int32_t>(), plan.indptr.data_ptr<int32_t>(),
        plan.last_page_len.data_ptr<int32_t>(), output.data_ptr(), batch_size,
        num_q_heads, num_kv_heads, page_size, dtype_code, stream);
    if (error) {
      throw std::runtime_error(error);
    }
    return new at::Tensor(std::move(output));
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" void* sglang_flashinfer_native_prefill_plan(
    void*, int32_t*, int32_t*, int32_t, int32_t, int32_t, int32_t, int32_t,
    void*, const char**);
extern "C" const char* sglang_flashinfer_native_prefill_run(
    const void*, void*, void*, void*, int32_t*, int32_t*, int32_t*, int32_t*,
    void*, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t,
    int32_t, bool, void*);

extern "C" void* sglang_flashinfer_prepare_prefill(
    void* existing, const at::Tensor* q_indptr, const at::Tensor* prefixes,
    const at::Tensor* block_table, int64_t q_heads, int64_t kv_heads,
    int64_t page_size, int64_t dtype) {
  try {
    if (!q_indptr->is_cuda() || q_indptr->scalar_type() != at::kInt ||
        q_indptr->dim() != 1 || !q_indptr->is_contiguous() ||
        !prefixes->is_cuda() || prefixes->scalar_type() != at::kInt ||
        prefixes->dim() != 1 || prefixes->size(0)+1 != q_indptr->size(0) ||
        prefixes->device() != q_indptr->device() ||
        block_table->device() != q_indptr->device() ||
        block_table->dim() != 2 || block_table->size(0) != prefixes->size(0))
      throw std::runtime_error("invalid FlashInfer prefill metadata");
    c10::cuda::CUDAGuard guard(q_indptr->device());
    auto q_host = q_indptr->to(at::kCPU);
    auto prefix_host = prefixes->to(at::kCPU);
    const auto batch = prefixes->size(0);
    auto* qp = q_host.data_ptr<int32_t>();
    auto* pp = prefix_host.data_ptr<int32_t>();
    bool paged = false;
    for (int64_t i = 0; i < batch; ++i) {
      if (pp[i] < 0 || qp[i+1] <= qp[i] || qp[0] != 0)
        throw std::runtime_error("invalid FlashInfer prefill lengths");
      if ((int64_t(pp[i]) + qp[i+1] - qp[i] + page_size - 1) / page_size > block_table->size(1))
        throw std::runtime_error("FlashInfer prefill block table is too small");
      paged |= pp[i] > 0;
    }
    auto created = existing ? nullptr : std::make_unique<FlashInferPlan>();
    auto* plan = existing ? static_cast<FlashInferPlan*>(existing) : created.get();
    if (existing && (plan->device_index != q_indptr->get_device() ||
        plan->num_q_heads != q_heads || plan->num_kv_heads != kv_heads ||
        plan->page_size != page_size || plan->dtype_code != dtype))
      throw std::runtime_error("FlashInfer prefill geometry changed");
    plan->q_indptr = *q_indptr;
    at::Tensor kv_host;
    if (paged) {
      auto lengths = prefixes->contiguous() + q_indptr->slice(0,1) - q_indptr->slice(0,0,-1);
      auto pages = at::floor_divide(lengths + page_size - 1, page_size);
      plan->indptr = at::cat({at::zeros({1}, pages.options()), pages.cumsum(0,at::kInt)});
      auto columns = at::arange(block_table->size(1), pages.options());
      plan->indices = block_table->masked_select(columns.unsqueeze(0) < pages.unsqueeze(1)).to(at::kInt).contiguous();
      plan->last_page_len = ((lengths - 1) % page_size + 1).contiguous();
      kv_host = plan->indptr.to(at::kCPU);
    } else {
      plan->indptr = *q_indptr;
      kv_host = q_host;
    }
    // The blocking metadata read drains prior work before reusing pinned storage.
    const char* error = nullptr;
    auto stream = c10::cuda::getCurrentCUDAStream(q_indptr->get_device()).stream();
    auto native = sglang_flashinfer_native_prefill_plan(plan->native_plan,
        qp, kv_host.data_ptr<int32_t>(), qp[batch], batch, q_heads, kv_heads,
        paged ? page_size : 1, stream, &error);
    if (!native) throw std::runtime_error(error ? error : "FlashInfer prefill plan failed");
    plan->native_plan = native;
    plan->prefill = true; plan->paged = paged; plan->total_queries = qp[batch];
    plan->num_q_heads = q_heads; plan->num_kv_heads = kv_heads;
    plan->head_dim = 128; plan->page_size = page_size; plan->dtype_code = dtype;
    plan->device_index = q_indptr->get_device();
    return existing ? existing : created.release();
  } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}

extern "C" at::Tensor* sglang_flashinfer_prefill(
    const void* prepared, const at::Tensor* q, const at::Tensor* k, const at::Tensor* v) {
  try {
    const auto& plan = *static_cast<const FlashInferPlan*>(prepared);
    c10::cuda::CUDAGuard guard(plan.device_index);
    const auto dtype = plan.dtype_code == 0 ? at::kBFloat16 : at::kHalf;
    for (auto tensor : {q,k,v}) {
      if (!tensor->is_cuda() || tensor->get_device() != plan.device_index ||
          tensor->scalar_type() != dtype || tensor->stride(-1) != 1 ||
          tensor->size(-1) != 128 || tensor->stride(-2) != 128)
        throw std::runtime_error("invalid FlashInfer prefill tensors");
    }
    if (!plan.prefill || q->dim()!=3 || q->size(0)!=plan.total_queries || q->size(1)!=plan.num_q_heads ||
        k->sizes()!=v->sizes() ||
        (plan.paged ? (k->dim()!=4 || !k->is_contiguous() || !v->is_contiguous() ||
            k->size(1)!=plan.page_size || k->size(2)!=plan.num_kv_heads)
         : (k->dim()!=3 || k->size(0)!=q->size(0) || k->size(1)!=plan.num_kv_heads)))
      throw std::runtime_error("FlashInfer prefill shape mismatch");
    auto out = at::empty(q->sizes(), q->options());
    auto stream = c10::cuda::getCurrentCUDAStream(plan.device_index).stream();
    auto error = sglang_flashinfer_native_prefill_run(plan.native_plan,q->data_ptr(),
        k->data_ptr(),v->data_ptr(),plan.q_indptr.data_ptr<int32_t>(),
        plan.indptr.data_ptr<int32_t>(),
        plan.paged ? plan.indices.data_ptr<int32_t>() : nullptr,
        plan.paged ? plan.last_page_len.data_ptr<int32_t>() : nullptr,
        out.data_ptr(),plan.q_indptr.size(0)-1,plan.num_q_heads,plan.num_kv_heads,
        plan.page_size,q->stride(0),k->stride(0),v->stride(0),plan.dtype_code,plan.paged,stream);
    if (error) throw std::runtime_error(error);
    return new at::Tensor(std::move(out));
  } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
