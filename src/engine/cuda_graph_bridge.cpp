#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <string>

struct GraphBridge {
  at::cuda::CUDAGraph graph;
  c10::cuda::CUDAStream previous;
  c10::cuda::CUDAStream capture;
  explicit GraphBridge(int device)
      : previous(c10::cuda::getCurrentCUDAStream(device)),
        capture(c10::cuda::getStreamFromPool(false, device)) {}
};

static thread_local std::string last_error;

extern "C" const char* sglang_graph_error() { return last_error.c_str(); }

extern "C" GraphBridge* sglang_graph_create(int device) {
  try {
    auto* bridge = new GraphBridge(device);
    c10::cuda::setCurrentCUDAStream(bridge->capture);
    return bridge;
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" bool sglang_graph_begin(GraphBridge* bridge) {
  try {
    bridge->graph.capture_begin();
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" bool sglang_graph_end(GraphBridge* bridge) {
  try {
    bridge->graph.capture_end();
    c10::cuda::setCurrentCUDAStream(bridge->previous);
    return true;
  } catch (const std::exception& error) {
    c10::cuda::setCurrentCUDAStream(bridge->previous);
    last_error = error.what();
    return false;
  }
}

extern "C" bool sglang_graph_replay(GraphBridge* bridge) {
  try {
    bridge->graph.replay();
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
    return false;
  }
}

extern "C" void sglang_graph_free(GraphBridge* bridge) {
  c10::cuda::setCurrentCUDAStream(bridge->previous);
  delete bridge;
}
