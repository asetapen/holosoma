#!/usr/bin/env python3
"""Convert a /pico_body_state MCAP into a holosoma reference-motion NPZ.

Produces the NPZ schema the policy's ``--task.ref-motion-path`` flag
expects: ``joint_pos`` (T, 36 = root_pos 3 + root_quat_wxyz 4 +
joints 29), ``joint_vel`` (T, 35), ``body_pos_w`` (T, B, 3),
``body_quat_w`` (T, B, 4) where B = len(G1_BODY_NAMES), plus
``fps`` / ``joint_names`` / ``body_names`` tags.

Pipeline:
    1. Decode /pico_body_state → (T, 24, 7) SMPL body joints
    2. Per-frame SMPLRetargeter → (q_joints[29], root_pos[3], root_quat_wxyz[4])
    3. Linearly interpolate to output_fps (default 50 Hz)
    4. MuJoCo FK over the G1 29-DOF scene → 32 body poses
    5. np.savez

Usage:
    python3 mcap_to_holosoma_npz.py \\
        --mcap path/to/pico_recording.mcap \\
        --out  path/to/ref_motion.npz
"""

from __future__ import annotations

import argparse
import os
import platform
import struct
import sys
from pathlib import Path


def _set_mujoco_gl_default() -> None:
    """Default MUJOCO_GL to egl (linux) / glfw (darwin) for offscreen rendering.

    Called from ``main()`` so importing this module as a library does not
    mutate the caller's environment.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")


# ---------------------------------------------------------------------------
# G1 body name table (matches holosoma NPZ convention)
# ---------------------------------------------------------------------------

G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def _decode_pico(cdr_buf: bytes):
    import numpy as np

    floats = struct.unpack_from("<168d", cdr_buf, 4)
    return np.asarray(floats, dtype=np.float64).reshape(24, 7)


def _load_mcap(mcap_path: Path):
    import numpy as np
    from mcap.reader import make_reader

    poses = []
    times = []
    with open(mcap_path, "rb") as f:
        r = make_reader(f)
        for _, _, msg in r.iter_messages(topics=["/pico_body_state"]):
            if len(msg.data) < 4 + 168 * 8:
                continue
            poses.append(_decode_pico(msg.data))
            times.append(msg.log_time)

    if not poses:
        raise RuntimeError(f"no /pico_body_state frames in {mcap_path}")

    # Sanity-check the first frame's pelvis quaternion. The decoder
    # assumes body_joint_poses is the first field in the CDR payload;
    # if the schema grows a preceding field we'd silently decode the
    # wrong floats and blow through the retargeter on garbage.
    pelvis_quat = poses[0][0, 3:7]
    quat_norm = float(np.linalg.norm(pelvis_quat))
    if not 0.9 < quat_norm < 1.1:
        raise RuntimeError(
            f"first-frame pelvis quaternion has norm {quat_norm:.3f}, expected ~1.0; "
            "the /pico_body_state CDR layout may have changed — the decoder "
            "assumes body_joint_poses is the first field."
        )
    return np.stack(poses, axis=0), np.asarray(times, dtype=np.int64)


def _retarget_sequence(poses, mjcf_path: str):
    """Run SMPLRetargeter on each frame; return (q[T,29], root_pos[T,3], root_quat_wxyz[T,4])."""
    import numpy as np
    from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter

    rt = SMPLRetargeter(mjcf_path, max_ik_iters=20)
    T = poses.shape[0]
    q = np.zeros((T, 29), dtype=np.float64)
    rp = np.zeros((T, 3), dtype=np.float64)
    rq = np.zeros((T, 4), dtype=np.float64)

    # Target pelvis height for the first frame. Use the retargeter's
    # neutral pelvis z (measured via FK at construction time against the
    # MJCF's home qpos) rather than a hardcoded 0.793 so this stays
    # consistent if the upstream MJCF is updated.
    target_pelvis_z = float(rt._g1_leg)

    z_offset = None
    for i in range(T):
        q_joints, _, _ = rt.retarget(poses[i])
        q[i] = q_joints
        rp[i] = rt.last_root_pos
        rq[i] = rt.last_root_quat_wxyz
        if z_offset is None:
            # One-shot lift: shift every frame by the same constant so
            # the first-frame pelvis sits at the neutral standing
            # height. This preserves motion dynamics (squats, steps) but
            # assumes the recording starts from a standing pose — if
            # the operator started crouched the whole clip rides high
            # until the next conversion pass.
            z_offset = target_pelvis_z - rp[i, 2]
        rp[i, 2] += z_offset
        if (i + 1) % 200 == 0 or i == T - 1:
            print(f"  retargeted {i + 1}/{T}", flush=True)
    return q, rp, rq


def _interp_to_fps(rp, rq_wxyz, q, in_stamps_ns, out_fps):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    # np.interp + Slerp require strictly-increasing xp. MCAP readers
    # iterate messages in chunk order, which is usually but not always
    # time-sorted. Sort + drop duplicate-timestamp frames.
    order = np.argsort(in_stamps_ns, kind="stable")
    in_stamps_ns = in_stamps_ns[order]
    rp = rp[order]
    rq_wxyz = rq_wxyz[order]
    q = q[order]
    unique_mask = np.concatenate(([True], np.diff(in_stamps_ns) > 0))
    in_stamps_ns = in_stamps_ns[unique_mask]
    rp = rp[unique_mask]
    rq_wxyz = rq_wxyz[unique_mask]
    q = q[unique_mask]

    t_in = (in_stamps_ns - in_stamps_ns[0]) / 1e9
    duration = float(t_in[-1])
    T_out = max(int(duration * out_fps) + 1, 2)
    t_out = np.linspace(0.0, duration, T_out)

    rp_out = np.zeros((T_out, 3))
    q_out = np.zeros((T_out, q.shape[1]))
    for j in range(3):
        rp_out[:, j] = np.interp(t_out, t_in, rp[:, j])
    for j in range(q.shape[1]):
        q_out[:, j] = np.interp(t_out, t_in, q[:, j])

    # SLERP for root orientation.
    rq_xyzw = rq_wxyz[:, [1, 2, 3, 0]]
    rots = Rotation.from_quat(rq_xyzw)
    slerp = Slerp(t_in, rots)
    rq_out_wxyz = slerp(t_out).as_quat(scalar_first=True)
    return rp_out, rq_out_wxyz, q_out


def _fk_bodies(rp, rq_wxyz, q, mjcf_path: str):
    """Run MuJoCo FK. Returns ``(body_pos_w, body_quat_w)`` arrays shaped
    ``(T, len(G1_BODY_NAMES), 3)`` and ``(T, len(G1_BODY_NAMES), 4)``."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    T = rp.shape[0]

    body_indices = []
    missing = []
    for n in G1_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid < 0:
            missing.append(n)
        else:
            body_indices.append(bid)
    if missing:
        # Silently falling back to body-id 0 (world) would emit
        # zero-vector body_pos_w / identity body_quat_w for the
        # missing links, which becomes garbage training data
        # downstream. Fail loudly — the caller should update
        # G1_BODY_NAMES or point at a matching MJCF.
        raise RuntimeError(
            f"MJCF at {mjcf_path} is missing required body links: {missing}. "
            "Update G1_BODY_NAMES in this script or pass --mjcf pointing "
            "at a MJCF that includes them."
        )

    n_bodies = len(G1_BODY_NAMES)
    bp = np.zeros((T, n_bodies, 3))
    bq = np.zeros((T, n_bodies, 4))
    for t in range(T):
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rq_wxyz[t]  # wxyz
        data.qpos[7 : 7 + q.shape[1]] = q[t]
        mujoco.mj_kinematics(model, data)
        bp[t] = data.xpos[body_indices]
        bq[t] = data.xquat[body_indices]
    return bp, bq


def main() -> int:
    _set_mujoco_gl_default()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--mjcf", type=Path, default=None,
                   help="Path to g1_29dof.xml (defaults to the shipped retargeter MJCF).")
    args = p.parse_args()

    import numpy as np

    if args.mjcf is None:
        import holosoma_retargeting as _hr
        args.mjcf = Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"

    print(f"Reading {args.mcap} ...", flush=True)
    poses, stamps = _load_mcap(args.mcap)
    print(f"  {poses.shape[0]} frames, {(stamps[-1] - stamps[0]) / 1e9:.1f}s "
          f"at ~{(poses.shape[0] - 1) / max((stamps[-1] - stamps[0]) / 1e9, 1e-6):.1f} Hz", flush=True)

    print(f"Retargeting → G1 29-DOF (MJCF {args.mjcf}) ...", flush=True)
    q, rp, rq = _retarget_sequence(poses, str(args.mjcf))

    print(f"Interpolating {poses.shape[0]} → {args.fps} Hz ...", flush=True)
    rp_out, rq_out_wxyz, q_out = _interp_to_fps(rp, rq, q, stamps, args.fps)
    T_out = rp_out.shape[0]
    print(f"  produced {T_out} frames at {args.fps} Hz", flush=True)

    print(f"Running MuJoCo FK for {len(G1_BODY_NAMES)} body poses ...", flush=True)
    bp, bq_wxyz = _fk_bodies(rp_out, rq_out_wxyz, q_out, str(args.mjcf))

    # Finite-difference velocities.
    # Compute dt from the actual interpolated time grid so clips whose
    # duration isn't an integer number of frames at the requested fps
    # don't silently produce ~0.1%-biased velocities.
    dt = float(stamps[-1] - stamps[0]) / 1e9 / max(T_out - 1, 1) if T_out > 1 else 1.0 / args.fps
    bp_lin_vel = np.gradient(bp, dt, axis=0)
    bq_xyzw = bq_wxyz[:, :, [1, 2, 3, 0]]
    from scipy.spatial.transform import Rotation
    ang_vel = np.zeros_like(bp)
    for b in range(bp.shape[1]):
        rots = Rotation.from_quat(bq_xyzw[:, b, :])
        for t in range(1, T_out - 1):
            drot = rots[t - 1].inv() * rots[t + 1]
            ang_vel[t, b] = drot.as_rotvec() / (2 * dt)
        ang_vel[0, b] = ang_vel[1, b]
        ang_vel[-1, b] = ang_vel[-2, b]

    joint_vel = np.gradient(q_out, dt, axis=0)

    # Pack in holosoma NPZ schema.
    joint_pos_h = np.concatenate([rp_out, rq_out_wxyz, q_out], axis=1)
    root_lin_vel = bp_lin_vel[:, 0, :]
    root_ang_vel = ang_vel[:, 0, :]
    joint_vel_h = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1)

    G1_JOINT_NAMES_29 = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    out = {
        "fps": np.array(args.fps),
        "joint_pos": joint_pos_h.astype(np.float64),
        "joint_vel": joint_vel_h.astype(np.float64),
        "body_pos_w": bp.astype(np.float64),
        "body_quat_w": bq_wxyz.astype(np.float64),
        "body_lin_vel_w": bp_lin_vel.astype(np.float64),
        "body_ang_vel_w": ang_vel.astype(np.float64),
        "joint_names": np.array(G1_JOINT_NAMES_29),
        "body_names": np.array(G1_BODY_NAMES),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"Saved: {args.out} ({T_out} frames, {T_out / args.fps:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
