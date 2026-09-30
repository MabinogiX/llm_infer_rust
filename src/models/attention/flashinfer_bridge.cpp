#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

struct FlashInferPlan {
  at::Tensor indices;
  at::Tensor indptr;
  at::Tensor last_page_len;
  at::Tensor request_indices;
  at::Tensor kv_tile_indices;
  at::Tensor kv_chunk_size;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
  int64_t page_size;
  int64_t dtype_code;
};

extern "C" const char* sglang_flashinfer_launch(
    const void* query, const void* key_cache, const void* value_cache,
    const int32_t* page_indices, const int32_t* page_indptr,
    const int32_t* page_last_len, const int32_t* request_indices,
    const int32_t* kv_tile_indices, const int32_t* kv_chunk_size,
    void* output, int32_t batch_size,
    int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t page_size, int32_t dtype_code, void* stream);
}  // namespace

extern "C" const char* sglang_flashinfer_error() { return last_error.c_str(); }

extern "C" void* sglang_flashinfer_prepare(
    const at::Tensor* block_table, const at::Tensor* sequence_lengths,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t page_size, int64_t dtype_code) {
  try {
    if (!block_table->is_cuda() || !sequence_lengths->is_cuda()) {
      throw std::runtime_error("FlashInfer metadata must be CUDA tensors");
    }
    if (block_table->dim() != 2 || sequence_lengths->dim() != 1 ||
        block_table->size(0) != sequence_lengths->size(0)) {
      throw std::runtime_error("invalid FlashInfer block table or lengths");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads ||
        head_dim != 128 || page_size <= 0 || (dtype_code != 0 && dtype_code != 1)) {
      throw std::runtime_error("unsupported FlashInfer attention configuration");
    }
    auto plan = std::make_unique<FlashInferPlan>();
    auto pages = at::floor_divide(sequence_lengths->to(at::kInt) + page_size - 1,
                                  page_size);
    plan->indptr = at::cat({at::zeros({1}, pages.options()), pages.cumsum(0, at::kInt)});
    auto columns = at::arange(block_table->size(1), pages.options());
    plan->indices = block_table->masked_select(columns.unsqueeze(0) < pages.unsqueeze(1))
                        .to(at::kInt)
                        .contiguous();
    plan->last_page_len = ((sequence_lengths->to(at::kInt) - 1) % page_size + 1).contiguous();
    plan->request_indices = at::arange(block_table->size(0), pages.options());
    plan->kv_tile_indices = at::zeros_like(plan->request_indices);
    plan->kv_chunk_size = at::ones({1}, pages.options());
    plan->num_q_heads = num_q_heads;
    plan->num_kv_heads = num_kv_heads;
    plan->head_dim = head_dim;
    plan->page_size = page_size;
    plan->dtype_code = dtype_code;
    return plan.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" void sglang_flashinfer_plan_drop(void* plan) {
  delete static_cast<FlashInferPlan*>(plan);
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
        dtype_code != plan.dtype_code || batch_size != plan.request_indices.size(0)) {
      throw std::runtime_error("FlashInfer decode tensors do not match the prepared batch");
    }
    auto output = at::empty_like(*query);
    auto stream = c10::cuda::getCurrentCUDAStream(query->get_device()).stream();
    const char* error = sglang_flashinfer_launch(
        query->data_ptr(), key_cache->data_ptr(), value_cache->data_ptr(),
        plan.indices.data_ptr<int32_t>(), plan.indptr.data_ptr<int32_t>(),
        plan.last_page_len.data_ptr<int32_t>(), plan.request_indices.data_ptr<int32_t>(),
        plan.kv_tile_indices.data_ptr<int32_t>(), plan.kv_chunk_size.data_ptr<int32_t>(),
        output.data_ptr(), batch_size,
        num_q_heads, num_kv_heads, head_dim, page_size, dtype_code, stream);
    if (error) {
      throw std::runtime_error(error);
    }
    return new at::Tensor(std::move(output));
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}
