#!/usr/bin/env python3
"""Replay fixed one-token streaming requests and fail on excessive TTFT.

Requires a running, idle service and the benchmark's workload.json. Use a
fresh server for cold probes; replaying the fixture again tests cached inputs.
This is an isolated performance probe, not a machine-independent CI threshold.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('pool_bench', Path(__file__).with_name('benchmark-shared-graph-pool.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://127.0.0.1:18115')
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--case', default='cold512')
parser.add_argument('--count', type=int, default=24)
parser.add_argument('--max-ttft-ms', type=float, default=30)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
rows = [row for trial in json.loads(args.workload.read_text()) for row in trial[args.case]['rows']]
if len(rows) < args.count + 2:
    parser.error('workload needs at least count+2 distinct requests')
bench.group(args.base_url, rows[:2], 1)
result = bench.group(args.base_url, rows[2:args.count + 2], 1)
result['threshold_ms'] = args.max_ttft_ms
result['slow_keys'] = [row['key'] for row in result['samples'] if row['ttft_ms'] > args.max_ttft_ms]
result['max_ttft_ms'] = max(row['ttft_ms'] for row in result['samples'])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'samples'}))
sys.exit(1 if result['slow_keys'] else 0)
