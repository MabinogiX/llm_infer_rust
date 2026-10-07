#include <cuda_runtime.h>
#include <flashinfer/sampling.cuh>

extern "C" const char* sglang_sampling_native(
    float* probs, int64_t* output, bool* valid, int32_t rows,
    int32_t vocab, int64_t top_k, float top_p, uint64_t seed, uint64_t offset, void* stream) {
  using namespace flashinfer::sampling;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t status;
  if (top_k < vocab) {
    status = TopKTopPSamplingFromProb<float,int64_t>(probs,nullptr,nullptr,
        output,valid,nullptr,rows,top_k,top_p,vocab,true,
        nullptr,seed,nullptr,offset,cuda_stream);
  } else if (top_p < 1.f) {
    status = TopPSamplingFromProb<float,int64_t>(probs,output,valid,nullptr,nullptr,
        rows,top_p,vocab,true,nullptr,seed,nullptr,offset,cuda_stream);
  } else {
    status = SamplingFromProb<float,int64_t>(probs,output,valid,nullptr,
        rows,vocab,true,nullptr,seed,nullptr,offset,cuda_stream);
  }
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}
