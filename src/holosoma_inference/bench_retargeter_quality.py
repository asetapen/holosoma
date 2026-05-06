#!/usr/bin/env python3
"""Diff retargeter output across max_ik_iters settings.

We dropped max_ik_iters from 20 (mink default) to 1 (--fast preset)
for a ~2x perf win. That was safe on the static parts of
pico_example_long (bench_retargeter_quality showed no change) but
what about full-body dynamic segments?

This script runs the shipped G1 retargeter over the entire MCAP at
iters in (1, 2, 4, 8, 20), captures q_joints per frame, then
reports per-joint max/mean absolute delta vs iters=20 (the nominal
converged reference). If delta < a few degrees, iters=1 is safe.

Usage:
    bazel run //...:bench_retargeter_quality -- \\
        --mcap $HOLOSOMA_TEST_DATA/pico_example_long.mcap \\
        --max-frames 1500
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def _load_poses(mcap_path: Path, max_frames: int):
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


def _run_iters(poses, ik_iters, mjcf: Path):
    import numpy as np
    from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter

    rt = SMPLRetargeter(str(mjcf), max_ik_iters=ik_iters)
    qs = np.zeros((len(poses), 29), dtype=np.float64)
    for i, p in enumerate(poses):
        q, _, _ = rt.retarget(p)
        qs[i] = q
    return qs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--iters", type=str, default="1,2,4,8,20")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    import numpy as np
    import holosoma_retargeting as _hr

    mjcf = Path(_hr.__file__).parent / "models" / "g1" / "g1_29dof.xml"

    poses = _load_poses(args.mcap, args.max_frames)
    print(f"loaded {len(poses)} frames from {args.mcap.name}")

    iter_list = [int(x) for x in args.iters.split(",")]
    print(f"iters sweep: {iter_list}")

    results: dict = {}
    for n in iter_list:
        print(f"  running iters={n}...", flush=True)
        qs = _run_iters(poses, n, mjcf)
        results[n] = qs

    # Reference: highest iter count.
    ref_iters = max(iter_list)
    ref = results[ref_iters]

    # Per-joint names.
    try:
        from holosoma_inference.config.config_values.robot import g1_29dof

        dof_names = g1_29dof.dof_names
    except Exception:
        dof_names = [f"j{i}" for i in range(29)]

    print()
    print(f"==== delta vs iters={ref_iters} ====")
    header = f"{'iters':>6s}  {'max|Δq|':>10s} {'mean|Δq|':>10s} {'max joint':>24s}"
    print(header)
    print("-" * len(header))
    summary = {"ref_iters": ref_iters, "per_iters": {}, "per_joint_max_deg": {}}
    for n in iter_list:
        if n == ref_iters:
            continue
        delta = results[n] - ref
        abs_d = np.abs(delta)
        max_d = abs_d.max()
        mean_d = abs_d.mean()
        # Joint of max diff
        max_j = int(np.unravel_index(abs_d.argmax(), abs_d.shape)[1])
        # Count frames where any joint differs by >5° — a noticeable
        # IK-minimum snap; useful to distinguish "sometimes wobbles"
        # from "fundamentally different trajectory."
        per_frame_max = abs_d.max(axis=1)  # (N,)
        frames_gt_5deg = int((per_frame_max > np.deg2rad(5)).sum())
        frames_gt_30deg = int((per_frame_max > np.deg2rad(30)).sum())
        print(
            f"{n:>6d}  {np.rad2deg(max_d):>8.2f}° {np.rad2deg(mean_d):>8.3f}° {dof_names[max_j]:>24s}"
            f"  |  frames>5°: {frames_gt_5deg}/{len(poses)} ({100 * frames_gt_5deg / len(poses):.1f}%)"
            f"  >30°: {frames_gt_30deg}/{len(poses)}"
        )
        summary["per_iters"][n] = {
            "max_abs_rad": float(max_d),
            "max_abs_deg": float(np.rad2deg(max_d)),
            "mean_abs_rad": float(mean_d),
            "mean_abs_deg": float(np.rad2deg(mean_d)),
            "max_joint": dof_names[max_j],
        }
        # Per-joint max |delta| in deg
        per_joint = np.rad2deg(abs_d.max(axis=0))
        summary["per_joint_max_deg"][str(n)] = {
            dof_names[i]: float(per_joint[i]) for i in range(29)
        }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
