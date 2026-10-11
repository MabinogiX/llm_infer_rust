"""Isolated allocator probe; no model or production graph implementation is changed.

Run each mode in a fresh process with CUDA_VISIBLE_DEVICES selecting an idle GPU.
"""

import argparse
import gc
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=(
    'private_different_streams', 'private_same_stream',
    'shared_different_streams', 'shared_same_stream'))
mode = parser.parse_args().mode
import torch

torch.cuda.set_device(0)
pool = None if mode.startswith('private') else torch.cuda.graph_pool_handle()
shared_stream = torch.cuda.Stream()
catalog = {}
sizes = [512, 256, 128]
layers = 8

def segment(x):
    scratch = x.repeat(1, 6)
    return scratch[:, :2560].clone()

for size in sizes:
    inputs = [torch.ones((size, 1024), device='cuda') for _ in range(layers)]
    records = []
    for index in range(layers):
        stream = torch.cuda.Stream() if mode.endswith('different_streams') else shared_stream
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = segment(inputs[index])
                del warm
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Match the Rust bridge's low-level begin/end; avoid the Python
        # context manager's extra per-capture empty_cache().
        with torch.cuda.stream(stream):
            if pool is None:
                graph.capture_begin()
            else:
                graph.capture_begin(pool=pool)
            output = segment(inputs[index])
            graph.capture_end()
        records.append((graph, output))
    catalog[size] = (inputs, records)

torch.cuda.synchronize()
torch.cuda.empty_cache()
held = {key: torch.cuda.memory_stats()[key] / 2**20 for key in (
    'allocated_bytes.all.current', 'reserved_bytes.all.current', 'active_bytes.all.current')}
checks = 0
# Different bucket order and eager seams; outputs stay strongly owned as in Rust.
for size in [128, 512, 256, 128, 256, 512]:
    inputs, records = catalog[size]
    inputs[0].fill_(size / 128)
    for index, (graph, output) in enumerate(records):
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, size / 128 * 2**index))
        checks += 1
        if index + 1 < layers:
            inputs[index + 1].copy_(output[:, :1024] * 2)
torch.cuda.synchronize()
del inputs, records, graph, output
catalog.clear()
gc.collect()
torch.cuda.empty_cache()
released = {key: torch.cuda.memory_stats()[key] / 2**20 for key in held}
print(json.dumps({'mode': mode, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
    'buckets': sizes, 'segments_per_bucket': layers, 'held_mib': held,
    'released_mib': released, 'output_checks': checks}))
