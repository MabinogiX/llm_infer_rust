#include <ATen/cuda/CUDAGraph.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <memory>

// A capture catalog belongs to one model/device. Its graphs retain this context;
// different model instances never implicitly share allocator storage.
struct GraphCaptureContext {
  int device;
  at::cuda::MempoolId_t pool;
  c10::cuda::CUDAStream capture;
  c10::cuda::CUDAStream execution;
  bool healthy = true;
  explicit GraphCaptureContext(int index)
      : device(index), pool(at::cuda::graph_pool_handle()),
        capture(c10::cuda::getStreamFromPool(false, index)),
        execution(c10::cuda::getCurrentCUDAStream(index)) {
    // A handle is only an ID. Keep the allocator pool live even if the first
    // capture is discarded before any successful graph enters the catalog.
    c10::cuda::CUDACachingAllocator::createOrIncrefPool(device, pool);
    // LibTorch also retains a pinned-host pool with this ID during capture.
    // begin/end acquires its reference without routing any host allocation here.
    try {
      auto* host = at::getHostAllocator(at::kCUDA);
      host->begin_allocate_to_pool(pool, [](c10::Stream) { return false; });
      host->end_allocate_to_pool(pool);
    } catch (...) {
      c10::cuda::CUDACachingAllocator::releasePool(device, pool);
      throw;
    }
  }
  ~GraphCaptureContext() {
    at::getHostAllocator(at::kCUDA)->release_pool(pool);
    c10::cuda::CUDACachingAllocator::releasePool(device, pool);
  }
};
using ContextHandle = std::shared_ptr<GraphCaptureContext>;

struct GraphBridge {
  ContextHandle context;
  at::cuda::CUDAGraph graph;
  c10::cuda::CUDAStream previous;
  bool capturing = false;
  bool stream_selected = true;
  explicit GraphBridge(ContextHandle owner)
      : context(std::move(owner)),
        previous(c10::cuda::getCurrentCUDAStream(context->device)) {}
  void restore_stream() {
    if (stream_selected) {
      c10::cuda::setCurrentCUDAStream(previous);
      stream_selected = false;
    }
  }
};

static thread_local std::string last_error;

extern "C" const char* sglang_graph_error() { return last_error.c_str(); }

extern "C" ContextHandle* sglang_graph_context_create(int device) {
  try {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    c10::cuda::CUDAGuard guard(device);
    return new ContextHandle(std::make_shared<GraphCaptureContext>(device));
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" void sglang_graph_context_free(ContextHandle* handle) { delete handle; }

extern "C" bool sglang_graph_context_check(ContextHandle* handle) {
  try {
    if (!(*handle)->healthy) {
      last_error = "shared CUDA graph capture context is invalid; rebuild the catalog";
      return false;
    }
    if (c10::cuda::getCurrentCUDAStream((*handle)->device) != (*handle)->execution) {
      last_error = "shared CUDA graphs require the catalog's execution stream";
      return false;
    }
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

// Device-wide allocator counters, for capture accounting and regression tests.
extern "C" bool sglang_graph_memory(int device, int64_t* allocated, int64_t* reserved) {
  try {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    *allocated = stats.allocated_bytes[0].current;
    *reserved = stats.reserved_bytes[0].current;
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

// Reset peaks at the catalog boundary, not for every bucket/segment. These
// counters cover the device allocator; direct CUDA allocations are excluded.
extern "C" bool sglang_graph_peak_memory(int device, bool reset,
                                        int64_t* allocated, int64_t* reserved) {
  try {
    if (reset) c10::cuda::CUDACachingAllocator::resetPeakStats(device);
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    *allocated = stats.allocated_bytes[0].peak;
    *reserved = stats.reserved_bytes[0].peak;
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" GraphBridge* sglang_graph_create(ContextHandle* handle) {
  try {
    if (!sglang_graph_context_check(handle)) return nullptr;
    auto bridge = std::make_unique<GraphBridge>(*handle);
    c10::cuda::setCurrentCUDAStream(bridge->context->capture);
    return bridge.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" bool sglang_graph_begin(GraphBridge* bridge) {
  try {
    c10::cuda::CUDAGuard guard(bridge->context->device);
    bridge->graph.capture_begin(bridge->context->pool);
    bridge->capturing = true;
    return true;
  } catch (const std::exception& error) {
    bridge->context->healthy = false;
    cudaStreamCaptureStatus status;
    if (cudaStreamIsCapturing(bridge->context->capture, &status) == cudaSuccess)
      bridge->capturing = status != cudaStreamCaptureStatusNone;
    last_error = error.what();
    return false;
  }
}

extern "C" bool sglang_graph_end(GraphBridge* bridge) {
  try {
    c10::cuda::CUDAGuard guard(bridge->context->device);
    bridge->capturing = false; // capture_end cleans allocator routing even on failure.
    bridge->graph.capture_end();
    bridge->restore_stream();
    return true;
  } catch (const std::exception& error) {
    bridge->context->healthy = false;
    bridge->restore_stream();
    last_error = error.what();
    return false;
  }
}

extern "C" bool sglang_graph_replay(GraphBridge* bridge) {
  try {
    if (!sglang_graph_context_check(&bridge->context)) return false;
    bridge->graph.replay();
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" void sglang_graph_free(GraphBridge* bridge) {
  c10::cuda::CUDAGuard guard(bridge->context->device);
  // Also covers unwinding while a Rust capture closure is running.
  if (bridge->capturing) {
    try { bridge->graph.capture_end(); }
    catch (...) { bridge->context->healthy = false; }
  }
  // Failed warmup/begin may leave the side stream selected. Destroying an
  // already completed graph must not change the caller's current stream.
  bridge->restore_stream();
  delete bridge;
}

// Metadata-only alias: unlike shallow_clone, this does not retain allocator
// storage. The caller must keep the graph pool live and enforce the value's
// capture/replay live range. Preserve strides and the view's offset data pointer.
extern "C" at::Tensor* sglang_graph_non_owning_view(const at::Tensor* tensor) {
  try {
    TORCH_CHECK(tensor->is_cuda(), "non-owning graph output requires CUDA");
    return new at::Tensor(at::from_blob(
        tensor->data_ptr(), tensor->sizes(), tensor->strides(), tensor->options()));
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}
