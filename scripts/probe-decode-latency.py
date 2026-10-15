#!/usr/bin/env python3
"""Measure warmed, serial decode latency through the streaming HTTP API.

Run on an idle server without a profiler. The optional TPOT limit is a local
performance assertion, not a hardware-independent CI threshold.
"""

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request


def measure(base_url, output_len):
    payload = {
        "model": "default",
        "prompt": "The sky is blue. " * 16,
        "temperature": 0,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    arrivals = []
    text = ""
    usage = None
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            fragment = choices[0].get("text", "") if choices else ""
            if fragment:
                arrivals.append(time.perf_counter())
                text += fragment
            if event.get("usage"):
                usage = event["usage"]
    end = time.perf_counter()
    if len(arrivals) < 2 or not usage:
        raise RuntimeError(f"expected streamed text and token usage: {len(arrivals)=}, {usage=}")
    tokens = usage["completion_tokens"]
    if tokens != output_len:
        raise RuntimeError(f"expected {output_len} tokens, received {tokens}")
    intervals = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "ttft_ms": (arrivals[0] - start) * 1000,
        "e2e_ms": (end - start) * 1000,
        "tpot_ms": (end - arrivals[0]) * 1000 / (tokens - 1),
        "median_itl_ms": statistics.median(intervals),
        "mean_itl_ms": statistics.mean(intervals),
        "chunks": len(arrivals),
        "usage": usage,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-tpot-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.output_len < 2:
        parser.error("samples >= 1, warmup >= 0 and output-len >= 2 are required")
    for _ in range(args.warmup):
        measure(args.base_url, args.output_len)
    rows = [measure(args.base_url, args.output_len) for _ in range(args.samples)]
    summary = {
        key: statistics.median(row[key] for row in rows)
        for key in ["ttft_ms", "e2e_ms", "tpot_ms", "median_itl_ms", "mean_itl_ms"]
    }
    summary["chunks"] = sorted({row["chunks"] for row in rows})
    summary["samples"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "requests": rows}, indent=2))
    print(json.dumps(summary), flush=True)
    if args.max_tpot_ms is not None and summary["tpot_ms"] > args.max_tpot_ms:
        raise SystemExit(f"FAIL: TPOT {summary['tpot_ms']:.3f} ms exceeds {args.max_tpot_ms} ms")


if __name__ == "__main__":
    main()
