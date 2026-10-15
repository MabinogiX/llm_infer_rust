#!/usr/bin/env python3
"""Alternate release binaries on one idle GPU with identical decode workloads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="append", required=True, help="LABEL=PATH; repeat for each stage")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    if args.rounds < 1 or args.samples < 1:
        parser.error("rounds and samples must be positive")
    binaries = [entry.split("=", 1) for entry in args.binary]
    if any(len(entry) != 2 or not Path(entry[1]).is_file() for entry in binaries):
        parser.error("every binary must be LABEL=PATH to an existing executable")
    if len({label for label, _ in binaries}) != len(binaries):
        parser.error("binary labels must be unique")
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("probe", root / "scripts/probe-decode-latency.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    torch_lib = next((root / ".venv/lib").glob("python*/site-packages/torch/lib"))
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "SGLANG_PROFILE_STEPS": "0",
        "PATH": f'{root}/.venv/bin:' + os.environ["PATH"],
        "LD_LIBRARY_PATH": f'/usr/local/cuda/compat:{torch_lib}:' + os.environ.get("LD_LIBRARY_PATH", ""),
        "LD_PRELOAD": str(torch_lib / "libtorch_cuda.so"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary_manifest = {
        label: {"path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for label, path in binaries
    }
    (args.output_dir / "binaries.json").write_text(json.dumps(binary_manifest, indent=2))
    base = f"http://127.0.0.1:{args.port}"
    results = []
    expected = {}
    for trial in range(args.rounds):
        order = binaries[trial % len(binaries):] + binaries[:trial % len(binaries)]
        for label, binary in order:
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    raise RuntimeError(f"test port already has a server: {base}")
            except OSError:
                pass
            log_path = args.output_dir / f"{trial}-{label}.log"
            cmd = [str(Path(binary).resolve()), "--model-path", str(args.model_path.resolve()),
                   "--device", "cuda", "--port", str(args.port), "--max-running-req", "4",
                   "--max-seq-len", "40960", "--attention-backend", "flashinfer"]
            print(f"START round={trial} stage={label} sha256={binary_manifest[label]['sha256']}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 180
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(log_path.read_text()[-5000:])
                        try:
                            with urllib.request.urlopen(base + "/health", timeout=1) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError("server startup timed out")
                        time.sleep(.2)
                    for length, concurrency in [(32, 1), (128, 1), (128, 4)]:
                        case = f"c{concurrency}-o{length}"
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            for _ in range(2):
                                list(pool.map(lambda _: probe.measure(base, length), range(concurrency)))
                            start = time.perf_counter()
                            rows = []
                            for _ in range(args.samples):
                                rows.extend(pool.map(lambda _: probe.measure(base, length), range(concurrency)))
                            elapsed = time.perf_counter() - start
                        hashes = sorted({hashlib.sha256(row["text"].encode()).hexdigest() for row in rows})
                        if case in expected and hashes != expected[case]:
                            raise RuntimeError(f"generated output differs for {label}/{case}: {hashes} != {expected[case]}")
                        expected[case] = hashes
                        for row in rows:
                            row["text_sha256"] = hashlib.sha256(row.pop("text").encode()).hexdigest()
                        result = dict(round=trial, stage=label, case=case,
                            median_tpot_ms=statistics.median(row["tpot_ms"] for row in rows),
                            median_itl_ms=statistics.median(row["median_itl_ms"] for row in rows),
                            median_ttft_ms=statistics.median(row["ttft_ms"] for row in rows),
                            output_tok_s=length * len(rows) / elapsed, requests=rows)
                        results.append(result)
                        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
                        print(json.dumps({k: v for k, v in result.items() if k != "requests"}), flush=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    print("PASS: all usage lengths and generated-text hashes match across stages", flush=True)


if __name__ == "__main__":
    main()
