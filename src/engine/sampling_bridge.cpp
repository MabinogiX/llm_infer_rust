#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace { thread_local std::string last_error; }
extern "C" const char* sglang_sampling_native(float*,int64_t*,bool*,int32_t,
    int32_t,int64_t,float,uint64_t,uint64_t,void*);
extern "C" const char* sglang_sampling_error() { return last_error.c_str(); }
extern "C" at::Tensor* sglang_sampling_flashinfer(const at::Tensor* probs, int64_t top_k, double top_p) {
  try {
    if (!probs->is_cuda() || probs->scalar_type()!=at::kFloat ||
        probs->dim()!=2 || !probs->is_contiguous() || top_p<=0 || top_p>1 ||
        top_k<1 || top_k>probs->size(1))
      throw std::runtime_error("invalid FlashInfer sampling probabilities");
    c10::cuda::CUDAGuard guard(probs->device());
    auto generator = at::cuda::detail::getDefaultCUDAGenerator(probs->get_device());
    auto* impl = generator.get<at::CUDAGeneratorImpl>();
    std::pair<uint64_t,uint64_t> rng;
    {
      std::lock_guard<std::mutex> lock(impl->mutex_);
      // One independent subsequence per row. Reserve a disjoint range even
      // for the rejection loop (at most vocab pivot refinements per sample).
      rng = impl->philox_engine_inputs(uint64_t(probs->size(1)) * 4 + 4);
    }
    auto output = at::empty({probs->size(0)},probs->options().dtype(at::kLong));
    auto valid = at::empty({probs->size(0)},probs->options().dtype(at::kBool));
    auto stream = c10::cuda::getCurrentCUDAStream(probs->get_device()).stream();
    auto error = sglang_sampling_native(probs->data_ptr<float>(),output.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),probs->size(0),probs->size(1),top_k,top_p,rng.first,rng.second,stream);
    if (error) throw std::runtime_error(error);
    // Retain validation without a device-to-host synchronization per group.
    // Invalid distributions produce -1 and are reported after the batch copy.
    output.masked_fill_(valid.logical_not(),-1);
    return new at::Tensor(std::move(output));
  } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
