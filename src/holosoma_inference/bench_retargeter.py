#!/usr/bin/env python3
"""Benchmark SMPLRetargeter.retarget() in isolation.

Background: holosoma driver's 'inference' latency (60-90 ms) is
NOT the ONNX model (that's ~0.3 ms). It's the SMPLRetargeter that runs
inside rl_inference. This bench measures retargeter latency directly so
we can iterate on ik_iters, posture cost weights, lm_damping, and the
mink Configuration refresh without spinning up the full stack.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def _dummy_pose(n_frames: int):
    """Manufacture a (n_frames, 24, 7) array of tracker-frame pico poses.

    Uses an identity-ish standing pose with a small time-varying perturbation
    on shoulder / elbow so the retargeter's IK actually has to work. Good
    enough to exercise the inner loop without needing a real MCAP here.
    """
    import numpy as np

    # Identity quats (xyzw) everywhere, joint positions on the Y-axis
    # (tracker Y-up) at plausible body-part heights.
    poses = np.zeros((n_frames, 24, 7), dtype=np.float64)
    poses[:, :, 6] = 1.0  # w=1 identity quat
    # Height per joint (meters above ground, roughly SMPLH defaults).
    heights = [
        0.95, 0.88, 0.88, 1.05, 0.48, 0.48, 1.15, 0.08, 0.08, 1.22,
        0.02, 0.02, 1.45, 1.40, 1.40, 1.55, 1.35, 1.35, 1.10, 1.10,
        0.85, 0.85, 0.85, 0.85,
    ]
    for j in range(24):
        poses[:, j, 1] = heights[j]
    # Sinusoidal perturbation on pelvis yaw axis to avoid static-IK caching.
    t = np.linspace(0, 2 * np.pi, n_frames)
    poses[:, 0, 0] = 0.05 * np.sin(t)
    return poses


def _load_mcap_poses(mcap_path, max_frames: int):
    """Decode PicoBodyState.body_joint_poses -> (N, 24, 7) numpy.

    The semantic layer of validate_pico_replay already does this; we
    duplicate the tiny decoder here to avoid a cross-package import and
    to keep this bench standalone.
    """
    import struct

    import numpy as np
    from mcap.reader import make_reader

    poses = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, msg in reader.iter_messages(topics=["/pico_body_state"]):
            buf = msg.data
            if len(buf) < 4 + 168 * 8:
                continue
            floats = struct.unpack_from("<168d", buf, 4)
            poses.append(np.asarray(floats, dtype=np.float64).reshape(24, 7))
            if len(poses) >= max_frames:
                break
    return np.stack(poses, axis=0) if poses else np.zeros((0, 24, 7))


def _bench(retargeter, poses, warmup: int):
    samples = []
    for i in range(warmup):
        retargeter.retarget(poses[i % len(poses)])
    for i in range(len(poses)):
        t0 = time.perf_counter()
        retargeter.retarget(poses[i])
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--mjcf", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    # Sweep a few values of max_ik_iters (SMPLRetargeter's inner loop).
    p.add_argument("--sweep-iters", type=str, default="1,4,8,20",
                   help="Comma-separated list of max_ik_iters values to test.")
    p.add_argument("--mcap", type=Path, default=None,
                   help="Drive the bench from real /pico_body_state frames in this MCAP "
                   "instead of the synthetic pose. Reveals IK cost on realistic input.")
    args = p.parse_args()

    try:
        import holosoma_retargeting  # noqa
        from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter
    except Exception as exc:
        print(f"error: could not import SMPLRetargeter: {exc!r}", file=sys.stderr)
        return 2

    mjcf = args.mjcf
    if mjcf is None:
        mjcf = Path(holosoma_retargeting.__file__).parent / "models" / "g1" / "g1_29dof.xml"
    if not mjcf.is_file():
        print(f"error: MJCF not found: {mjcf}", file=sys.stderr)
        return 2

    if args.mcap is not None:
        poses = _load_mcap_poses(args.mcap, args.iters)
        print(f"  using real MCAP poses from {args.mcap.name} ({len(poses)} frames)")
    else:
        poses = _dummy_pose(args.iters)
    sweep = [int(x) for x in args.sweep_iters.split(",") if x.strip()]

    print(f"==== bench_retargeter: {mjcf.name}")
    print(f"  iters:    {args.iters}  warmup: {args.warmup}")
    print(f"  sweep:    max_ik_iters = {sweep}")
    print()
    header = f"{'max_ik_iters':>15s}  {'mean':>8s} {'p50':>8s} {'p99':>8s}   {'Hz_p50':>7s} {'Hz_p99':>7s}"
    print(header)
    print("-" * len(header))

    results = []
    for n_iters in sweep:
        try:
            rt = SMPLRetargeter(str(mjcf), max_ik_iters=n_iters)
            stats = _bench(rt, poses, args.warmup)
        except Exception as exc:
            print(f"{n_iters:>15d}  ERROR: {exc!r}")
            results.append({"max_ik_iters": n_iters, "error": repr(exc)})
            continue
        hz_p50 = 1000.0 / max(stats["p50_ms"], 1e-6)
        hz_p99 = 1000.0 / max(stats["p99_ms"], 1e-6)
        print(
            f"{n_iters:>15d}  {stats['mean_ms']:>6.2f}ms {stats['p50_ms']:>6.2f}ms {stats['p99_ms']:>6.2f}ms   {hz_p50:>6.0f} {hz_p99:>6.0f}"
        )
        results.append({"max_ik_iters": n_iters, **stats, "hz_p50": hz_p50, "hz_p99": hz_p99})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"mjcf": str(mjcf), "results": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
