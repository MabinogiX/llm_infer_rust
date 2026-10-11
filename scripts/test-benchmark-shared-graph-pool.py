#!/usr/bin/env python3
"""CPU-only regression tests for benchmark dependency selection and resume."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pool_bench', Path(__file__).with_name('benchmark-shared-graph-pool.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkDependencies(unittest.TestCase):
    def test_only_selected_binaries_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'target/release/sglang-rust'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'current')
            args = SimpleNamespace(repo=root, stage2_binary=root / 'missing-stage2',
                                   baseline_binary=root / 'missing-baseline', modes=['rust', 'eager'])
            self.assertEqual(bench.binary_hashes(args),
                             {'rust_binary': hashlib.sha256(b'current').hexdigest()})
            args.modes = ['sglang']
            self.assertEqual(bench.binary_hashes(args), {})
            args.modes = ['rust-stage2']
            with self.assertRaises(FileNotFoundError):
                bench.binary_hashes(args)

    def test_completed_resume_needs_no_binary_gpu_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = dict(model='/missing/model', gpu=1, rounds=1, modes=['eager'],
                            cases=bench.CASES, buckets=bench.BUCKETS,
                            nvml_v1_includes_driver_reserved_mib=869, rust_binary='recorded')
            (output / 'metadata.json').write_text(json.dumps(metadata))
            (output / 'workload.json').write_text(json.dumps([{}]))
            result = dict(mode='eager', trial=0, startup_s=1, ready_gpu_bytes=900 * 2**20,
                          steady_gpu_bytes=900 * 2**20, capture_log=[],
                          cases={case: dict(ttft_ms=1, e2e_ms=2, tpot_ms=None,
                                           output_tokens_per_s=3, samples=[])
                                 for case, *_ in bench.CASES})
            (output / '0-eager.json').write_text(json.dumps(result))
            original_metadata = (output / 'metadata.json').read_bytes()
            original_workload = (output / 'workload.json').read_bytes()
            argv = ['bench', '--output-dir', str(output), '--model', '/missing/model',
                    '--repo', '/missing/repo', '--rounds', '1', '--modes', 'eager', '--resume']
            with patch.object(sys, 'argv', argv), \
                 patch.object(bench.subprocess, 'check_output', side_effect=AssertionError('external dependency')), \
                 patch.object(bench, 'make_workload', side_effect=AssertionError('tokenizer')), \
                 contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            self.assertEqual((output / 'metadata.json').read_bytes(), original_metadata)
            self.assertEqual((output / 'workload.json').read_bytes(), original_workload)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['eager']['startup_s'], 1)


if __name__ == '__main__':
    unittest.main()
