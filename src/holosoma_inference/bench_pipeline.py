#!/usr/bin/env python3
"""Full-pipeline perf bench — approximate the driver's in-process load.

The standalone bench_retargeter shows SMPLRetargeter at ~4ms p50, the
bench_onnx_inference shows the WBT ONNX at ~0.3ms p50. But in-driver
steady-state the retargeter alone hits 60-90ms. Hypothesis: the driver
runs retargeter + ONNX + MujocoInterface PD loop concurrently in one
process, and something in that combination (threadpool contention, GIL,
or an ABI issue) inflates per-call latency ~18x.

This script reproduces the driver's *process-level* load without rclpy:
    1. Loads /pico_body_state frames from an MCAP.
    2. Builds a real SMPLRetargeter and a real MujocoInterface backend.
    3. On each tick: retarget -> (fake policy network time) -> send PD
       command through MujocoInterface -> advance sim.
    4. Measures per-stage latency over ``--iters`` ticks.

If this reproduces the slowdown, we've confirmed the integration is the
cause (rather than any one component). Then we can attribute to
retarget-vs-sim MuJoCo contention.

Usage (inside bazel hermetic env):
    bazel run //...:bench_pipeline -- \\
        --mcap $HOLOSOMA_TEST_DATA/pico_example_long.mcap \\
        --iters 300

    # Isolate the retargeter (no concurrent sim tick)
    bazel run //...:bench_pipeline -- --mcap <...> --no-sim

    # Mock the sim with a zero-op backend
    bazel run //...:bench_pipeline -- --mcap <...> --mock-backend
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path


def _load_mcap_poses(mcap_path: Path, max_frames: int):
    import numpy as np
    from mcap.reader import make_reader

    poses = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, _, msg in reader.iter_messages(topics=["/pico_body_state"]):
            buf = msg.data
            if len(buf) < 4 + 168 * 8:
                continue
            floats = struct.unpack_from("<168d", buf, 4)
            poses.append(np.asarray(floats, dtype=np.float64).reshape(24, 7))
            if len(poses) >= max_frames:
                break
    return np.stack(poses, axis=0) if poses else np.zeros((0, 24, 7))


def _stats(samples):
    if not samples:
        return {"iters": 0}
    s = sorted(samples)
    n = len(s)
    return {
        "iters": n,
        "mean_ms": statistics.fmean(s),
        "stdev_ms": statistics.pstdev(s),
        "min_ms": s[0],
        "p50_ms": s[n // 2],
        "p90_ms": s[int(n * 0.90)],
        "p99_ms": s[int(n * 0.99)],
        "max_ms": s[-1],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, required=True)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--ik-iters", type=int, default=4)
    p.add_argument("--no-sim", action="store_true",
                   help="Skip the MujocoInterface send_low_command — isolates the "
                   "retargeter latency in a realistic concurrent-subscriber process.")
    p.add_argument("--mock-backend", action="store_true",
                   help="Use a no-op backend (not MujocoInterface). Confirms whether "
                   "the concurrent sim is the contention source.")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    if not args.mcap.is_file():
        print(f"error: mcap not found: {args.mcap}", file=sys.stderr)
        return 2

    # --- Imports: heavyweight so the perf of 'import' isn't measured ---
    import numpy as np
    import mujoco  # noqa
    from dataclasses import replace as _replace

    from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter
    from holosoma_inference.config.config_values.robot import g1_29dof
    from holosoma_inference.sdk.mujoco.mujoco_interface import MujocoInterface

    # --- Build the components ---
    import holosoma_retargeting as _hr

    mjcf = Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"
    retargeter = SMPLRetargeter(str(mjcf), max_ik_iters=args.ik_iters)

    # MujocoInterface needs motor_kp/kd — G1 config has None in some drops;
    # synthesise benign values for the PD.
    cfg = g1_29dof
    if cfg.motor_kp is None or cfg.motor_kd is None:
        cfg = _replace(
            cfg,
            motor_kp=tuple([100.0] * cfg.num_motors),
            motor_kd=tuple([5.0] * cfg.num_motors),
        )

    import os as _os

    _os.environ["HOLOSOMA_MUJOCO_REAL_TIME"] = "0"  # deterministic sim stepping
    backend = None
    if not args.mock_backend:
        backend = MujocoInterface(cfg, use_joystick=False)

    poses = _load_mcap_poses(args.mcap, args.iters + args.warmup)
    print(f"==== bench_pipeline")
    print(f"  mcap:      {args.mcap.name} ({len(poses)} frames)")
    print(f"  ik_iters:  {args.ik_iters}")
    print(f"  sim:       {'off' if args.no_sim else 'on'}  mock_backend={args.mock_backend}")
    print()

    lat_retarget = []
    lat_sim = []
    lat_total = []

    for i in range(min(len(poses), args.iters + args.warmup)):
        pose = poses[i]
        t_total = time.perf_counter()

        # Stage 1: retarget
        t0 = time.perf_counter()
        q_joints, _, _ = retargeter.retarget(pose)
        lat_retarget.append((time.perf_counter() - t0) * 1000.0)

        # Stage 2: concurrent sim tick (what MujocoInterface would do
        # each time send_low_command runs)
        if backend is not None and not args.no_sim:
            t0 = time.perf_counter()
            backend.send_low_command(
                cmd_q=q_joints.astype(np.float64),
                cmd_dq=np.zeros(g1_29dof.num_joints),
                cmd_tau=np.zeros(g1_29dof.num_joints),
            )
            backend.step(10)  # 10 * 0.002s = 20ms of sim per 50Hz tick
            lat_sim.append((time.perf_counter() - t0) * 1000.0)
        else:
            lat_sim.append(0.0)

        lat_total.append((time.perf_counter() - t_total) * 1000.0)

    # Drop warmup
    lat_retarget = lat_retarget[args.warmup:]
    lat_sim = lat_sim[args.warmup:]
    lat_total = lat_total[args.warmup:]

    stats = {
        "retarget": _stats(lat_retarget),
        "sim_step": _stats(lat_sim),
        "total": _stats(lat_total),
    }

    header = f"{'stage':>10s}  {'mean':>8s} {'p50':>8s} {'p99':>8s}   {'Hz_p50':>7s} {'Hz_p99':>7s}"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        if s.get("iters", 0) == 0:
            continue
        hz_p50 = 1000.0 / max(s["p50_ms"], 1e-6)
        hz_p99 = 1000.0 / max(s["p99_ms"], 1e-6)
        print(
            f"{name:>10s}  {s['mean_ms']:>6.2f}ms {s['p50_ms']:>6.2f}ms {s['p99_ms']:>6.2f}ms   {hz_p50:>6.0f} {hz_p99:>6.0f}"
        )

    # Check: does 'total' exceed a 250Hz budget (4ms)?
    budget_ms = 4.0
    breaching = stats["total"]["p99_ms"] > budget_ms if stats["total"].get("iters") else False
    print()
    print(f"  250Hz budget: {budget_ms} ms/tick (p99) — {'OVER BUDGET' if breaching else 'OK'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=2))
        print(f"\n  wrote {args.json}")

    return 1 if breaching else 0


if __name__ == "__main__":
    sys.exit(main())
