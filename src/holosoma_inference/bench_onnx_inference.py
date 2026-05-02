#!/usr/bin/env python3
"""Benchmark ONNX inference latency for the holosoma WBT policy.

Why this exists: the holosoma driver today reports RL FPS of ~8-11 Hz on
pico_example_long with inference_time ~60-90 ms. Target control rate is
250 Hz, so inference needs to finish in under 4 ms. This script isolates
the InferenceSession from the rest of the pipeline so we can iterate on
session options / providers / intra-op threads without spinning up ROS.

Usage:
    # Default model, 1000 iters, all ort configurations we know how to try
    bench_onnx_inference.py

    # Specific model + JSON output
    bench_onnx_inference.py \\
        --model holosoma_extensions/models/active.onnx \\
        --iters 2000 --json /tmp/bench.json

Compares a grid of SessionOptions:
    * default (whatever onnxruntime picks — reproduces the baseline)
    * graph_opt=ALL
    * intra_op_num_threads in {1, 2, 4, 8, num_cores}
    * execution_mode in {SEQUENTIAL, PARALLEL}
    * providers available on this host (CPU, CUDA, TensorRT — whichever
      are compiled into the installed onnxruntime)

Prints one row per config with mean/p50/p99 latency and effective Hz.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _make_dummy_feed(session) -> dict:
    """Build a zero-filled input_feed matching the session's declared inputs."""
    import numpy as np

    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = np.float32 if "float" in inp.type else np.int64
        feed[inp.name] = np.zeros(shape, dtype=dtype)
    return feed


def _bench_one(session, feed, output_names, iters: int, warmup: int = 20):
    """Run `iters` sess.run calls and return latency stats in ms."""
    # Warmup (first call allocates weights + kernel caches).
    for _ in range(warmup):
        session.run(output_names, feed)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(output_names, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    n = len(samples)
    return {
        "iters": n,
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.pstdev(samples),
        "min_ms": samples[0],
        "p50_ms": samples[n // 2],
        "p90_ms": samples[int(n * 0.90)],
        "p99_ms": samples[int(n * 0.99)],
        "max_ms": samples[-1],
    }


def _configs_to_try():
    """Return a list of (label, session_options_fn, providers) triples.

    ``session_options_fn`` takes nothing and returns a fresh SessionOptions,
    so each bench run gets its own session state.
    """
    import onnxruntime as ort

    n_cores = os.cpu_count() or 4
    available = ort.get_available_providers()

    def _opts(intra: int, inter: int, graph_opt, exec_mode):
        def make():
            o = ort.SessionOptions()
            o.intra_op_num_threads = intra
            o.inter_op_num_threads = inter
            o.graph_optimization_level = graph_opt
            o.execution_mode = exec_mode
            return o

        return make

    configs = [
        # Baseline: whatever defaults onnxruntime picks if you pass nothing.
        ("default", lambda: None, None),
        # Single-threaded — eliminates threadpool contention as a suspect.
        (
            "intra=1,seq,opt=ALL",
            _opts(1, 1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
            ["CPUExecutionProvider"],
        ),
        (
            "intra=2,seq,opt=ALL",
            _opts(2, 1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
            ["CPUExecutionProvider"],
        ),
        (
            "intra=4,seq,opt=ALL",
            _opts(4, 1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
            ["CPUExecutionProvider"],
        ),
        (
            f"intra={n_cores},seq,opt=ALL",
            _opts(n_cores, 1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
            ["CPUExecutionProvider"],
        ),
        (
            "intra=4,par,opt=ALL",
            _opts(4, 2, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_PARALLEL),
            ["CPUExecutionProvider"],
        ),
        # No optimization (sanity check that ALL is actually helping).
        (
            "intra=1,seq,opt=NONE",
            _opts(1, 1, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
            ["CPUExecutionProvider"],
        ),
    ]

    # If CUDA/TensorRT are compiled in, try them too.
    if "CUDAExecutionProvider" in available:
        configs.append(
            (
                "CUDA,intra=1,seq,opt=ALL",
                _opts(1, 1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        )
    if "TensorrtExecutionProvider" in available:
        configs.append(
            (
                "TRT,intra=1,seq,opt=ALL",
                _opts(1, 1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.ExecutionMode.ORT_SEQUENTIAL),
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        )
    return configs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "active.onnx",
    )
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    import onnxruntime as ort

    print(f"==== bench_onnx_inference: {args.model.name}")
    print(f"  onnxruntime: {ort.__version__}")
    print(f"  providers:   {ort.get_available_providers()}")
    print(f"  cpu_count:   {os.cpu_count()}")
    print(f"  iters:       {args.iters} (warmup {args.warmup})")
    print()

    results: list[dict] = []
    header = f"{'config':38s}  {'mean':>7s} {'p50':>7s} {'p99':>7s} {'max':>7s}   {'Hz_p50':>7s} {'Hz_p99':>7s}"
    print(header)
    print("-" * len(header))

    for label, opts_fn, providers in _configs_to_try():
        try:
            opts = opts_fn()
            session = ort.InferenceSession(
                str(args.model),
                sess_options=opts,
                providers=providers,
            )
            feed = _make_dummy_feed(session)
            output_names = [o.name for o in session.get_outputs()]
            stats = _bench_one(session, feed, output_names, args.iters, args.warmup)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:38s}  ERROR: {exc!r}")
            results.append({"label": label, "error": repr(exc)})
            continue
        hz_p50 = 1000.0 / max(stats["p50_ms"], 1e-6)
        hz_p99 = 1000.0 / max(stats["p99_ms"], 1e-6)
        print(
            f"{label:38s}  {stats['mean_ms']:>6.2f}ms {stats['p50_ms']:>6.2f}ms {stats['p99_ms']:>6.2f}ms {stats['max_ms']:>6.2f}ms   {hz_p50:>6.0f} {hz_p99:>6.0f}"
        )
        results.append(
            {
                "label": label,
                "providers": providers or ["default"],
                **stats,
                "hz_p50": hz_p50,
                "hz_p99": hz_p99,
            }
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": str(args.model),
                    "onnxruntime": ort.__version__,
                    "available_providers": ort.get_available_providers(),
                    "iters": args.iters,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
