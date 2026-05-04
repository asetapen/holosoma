#!/usr/bin/env python3
"""Real-time pico-skeleton / retargeted-G1 viewer.

Two input modes:

  --mcap PATH          Replay a ``/pico_body_state`` MCAP at its natural
                       rate (defaults to 50 Hz if the MCAP doesn't carry
                       timing).

  --live               Subscribe to a live ``/pico_body_state`` ROS topic
                       via rclpy. Requires the bazel-hermetic env's
                       ``@ros2_rclpy``.

Display options:

  --show skeleton      Just the SMPL-24 stick figure (default).
  --show retarget      Skeleton + retargeted G1 (two windows).
  --show wbt           Skeleton + retargeted + WBT policy dof_pos
                       (three windows).

No video output — interactive 3D via ``mujoco.viewer.launch_passive``.
Usage (inside sim container):

    bazel run //holosoma_extensions/thirdparty/holosoma/src/holosoma_inference:pico_live_viewer \\
        -- --mcap path/to/recording.mcap

    # Live (needs ROS2 on PATH):
    bazel run //...:pico_live_viewer -- --live
"""

from __future__ import annotations

import argparse
import os
import platform
import struct
import sys
import time
from pathlib import Path


def _set_mujoco_gl_default() -> None:
    # Viewer needs a windowing GL backend, not egl. glfw works on both
    # linux (with DISPLAY) and darwin; egl is offscreen only.
    os.environ.setdefault("MUJOCO_GL", "glfw")


def _decode_pico(cdr_buf: bytes):
    import numpy as np
    floats = struct.unpack_from("<168d", cdr_buf, 4)
    return np.asarray(floats, dtype=np.float64).reshape(24, 7)


# SMPL-24 parent indices — same as three_panel_eval.py. Duplicated here
# so the viewer is usable standalone without pulling in that module.
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]


# ---------------------------------------------------------------------------
# MCAP replay source
# ---------------------------------------------------------------------------


def _mcap_source(mcap_path: Path, max_frames: int | None = None):
    """Yield ``(pose_24x7, log_time_ns)`` tuples from the MCAP in order."""
    from mcap.reader import make_reader

    paths: list[Path]
    if mcap_path.is_dir():
        paths = sorted(mcap_path.glob("*.mcap"))
        if not paths:
            raise FileNotFoundError(f"no *.mcap files in {mcap_path}")
    else:
        paths = [mcap_path]

    count = 0
    for mp in paths:
        with open(mp, "rb") as f:
            reader = make_reader(f)
            for _, _, msg in reader.iter_messages(topics=["/pico_body_state"]):
                if len(msg.data) < 4 + 168 * 8:
                    continue
                yield _decode_pico(msg.data), int(msg.log_time)
                count += 1
                if max_frames is not None and count >= max_frames:
                    return


# ---------------------------------------------------------------------------
# Live ROS source
# ---------------------------------------------------------------------------


def _live_source(topic: str = "/pico_body_state"):
    """Yield ``(pose_24x7, stamp_ns)`` from a live ROS subscription.

    Uses rclpy — must be run inside a ROS2 environment. The bazel
    hermetic env provides it via ``@ros2_rclpy``.
    """
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    # Inline import of PicoBodyState — the generated message lives in
    # the ROS2 build tree. Expect it on PYTHONPATH in the bazel env.
    try:
        from gmp_interfaces.msg import PicoBodyState
    except Exception as exc:
        raise RuntimeError(
            "Could not import gmp_interfaces.msg.PicoBodyState. "
            "Source a ROS2 env before running --live: "
            "`pi source ros && pi build`."
        ) from exc

    rclpy.init()
    # Shared mutable latest-frame slot. rclpy callbacks fire on an
    # executor thread; we copy out in the main loop.
    latest: dict[str, object | None] = {"pose": None, "stamp": None}

    class _Subscriber(Node):
        def __init__(self):
            super().__init__("pico_live_viewer")
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(PicoBodyState, topic, self._cb, qos)

        def _cb(self, msg):
            # body_joint_poses is a flat list of 168 doubles.
            arr = np.asarray(msg.body_joint_poses, dtype=np.float64).reshape(24, 7)
            stamp = msg.timestamp_ns if hasattr(msg, "timestamp_ns") else int(
                time.time_ns())
            latest["pose"] = arr
            latest["stamp"] = stamp

    node = _Subscriber()
    from rclpy.executors import SingleThreadedExecutor
    exe = SingleThreadedExecutor()
    exe.add_node(node)

    import threading
    spinner = threading.Thread(target=exe.spin, daemon=True)
    spinner.start()

    try:
        last_stamp = None
        while True:
            p = latest["pose"]
            s = latest["stamp"]
            if p is not None and s != last_stamp:
                last_stamp = s
                yield p, s
            else:
                time.sleep(0.002)
    finally:
        exe.shutdown()
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Scene construction: a minimal MJCF with the skeleton + optional G1.
# ---------------------------------------------------------------------------


def _build_skeleton_mjcf() -> str:
    """Return an MJCF string with 24 small spheres as mocap bodies so the
    viewer can show the skeleton alongside a G1 without rebuilding the
    scene each frame.
    """
    meshes = []
    bodies = []
    sites = []
    for i in range(24):
        bodies.append(
            f'    <body name="smpl_joint_{i}" pos="0 0 0" mocap="true">\n'
            f'      <geom type="sphere" size="0.03" rgba="1 0.7 0.4 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>'
        )
    body_xml = "\n".join(bodies)
    return f"""<?xml version="1.0"?>
<mujoco model="pico_skeleton">
  <worldbody>
    <geom type="plane" size="4 4 0.1" rgba="0.2 0.3 0.4 1"/>
    <light pos="0 0 4" castshadow="true"/>
{body_xml}
  </worldbody>
</mujoco>
"""


def _build_combined_mjcf(g1_xml_path: str) -> str:
    """Embed the G1 MJCF inside a scene that also contains the 24 SMPL
    mocap spheres + connecting line geoms. Returns an MJCF *string* so
    ``from_xml_string`` can load it without us reimplementing mesh path
    resolution (we still rely on the G1's ``meshdir`` pointing at the
    right asset directory — caller should pass an MJCF whose meshdir
    resolves from its own location).
    """
    # The simplest robust way: include the g1 MJCF verbatim, add our
    # skeleton bodies into a sibling <worldbody>. MuJoCo's `<include>`
    # doesn't work cross-directory from a string, so we inline the g1
    # and splice spheres into the existing <worldbody>.
    with open(g1_xml_path, "r", encoding="utf-8") as f:
        g1 = f.read()

    # Build the mocap bodies to splice in.
    mocap_bodies = "\n".join(
        f'    <body name="smpl_joint_{i}" pos="0 0 0" mocap="true">'
        f'<geom type="sphere" size="0.025" rgba="1 0.7 0.4 0.9" '
        f'contype="0" conaffinity="0"/></body>'
        for i in range(24)
    )

    # Splice right after the opening <worldbody> tag. This is robust to
    # G1 variants as long as they have a worldbody.
    if "<worldbody>" not in g1:
        raise RuntimeError(
            f"G1 MJCF {g1_xml_path} has no <worldbody> — cannot splice.")
    return g1.replace("<worldbody>", "<worldbody>\n" + mocap_bodies, 1)


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------


def _find_g1_mjcf() -> Path:
    import holosoma_retargeting as _hr
    return Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"


def _run_viewer(source_iter, show: str, real_time: bool):
    """Drive a mujoco.viewer.launch_passive session from the source iterator.

    ``show``: "skeleton" | "retarget" | "wbt" (wbt not yet implemented
    for live; uses the retargeter's own output as a proxy).
    """
    import numpy as np
    import mujoco
    import mujoco.viewer

    # Build scene. For skeleton-only, use a bare MJCF with just the
    # 24 mocap spheres. For retarget (and wbt), embed the G1 so we can
    # drive its qpos alongside the skeleton.
    if show == "skeleton":
        xml = _build_skeleton_mjcf()
        model = mujoco.MjModel.from_xml_string(xml)
    else:
        g1 = _find_g1_mjcf()
        # from_xml_path so relative meshdir resolves next to the XML.
        # We splice our mocap bodies by writing a temp file next to the
        # G1 (so assets/ resolves), then loading.
        combined = _build_combined_mjcf(str(g1))
        tmp = g1.parent / "_pico_live_tmp.xml"
        tmp.write_text(combined)
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp))
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    data = mujoco.MjData(model)

    # Index lookups for the 24 mocap bodies.
    mocap_body_ids = []
    for i in range(24):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"smpl_joint_{i}")
        if bid < 0:
            raise RuntimeError(f"scene missing smpl_joint_{i}")
        mocap_body_ids.append(bid)
    # Map each body_id -> mocap_id (0..n_mocap-1)
    mocap_ids = [int(model.body_mocapid[bid]) for bid in mocap_body_ids]

    # Build retargeter if requested.
    retargeter = None
    g1_dof_qpos_idx: list[int] = []
    if show in ("retarget", "wbt"):
        from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter
        retargeter = SMPLRetargeter(str(_find_g1_mjcf()), max_ik_iters=4)
        # Resolve the 29 joint qpos indices in the combined model.
        dof_names = list(retargeter._mj_model.joint(j).name
                         for j in range(retargeter._mj_model.njnt)
                         if retargeter._mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE)
        for name in dof_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"combined scene missing joint {name}")
            g1_dof_qpos_idx.append(int(model.jnt_qposadr[jid]))
        # G1 freejoint is [0:7] in the combined scene; start standing.
        if model.nq >= 7:
            data.qpos[0:3] = [0.0, 0.0, 0.793]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    print(f"Starting viewer (show={show}, real_time={real_time})", flush=True)
    last_stamp_ns: int | None = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for pose, stamp_ns in source_iter:
            if not viewer.is_running():
                break

            # Apply coord transform pico Y-up → robot Z-up to match the
            # retargeter's convention. Position-only — we only place the
            # mocap spheres.
            xyz = np.stack([pose[:, 0], -pose[:, 2], pose[:, 1]], axis=1)
            # Pin pelvis planar; keep height.
            xyz[:, 0] -= xyz[0, 0]
            xyz[:, 1] -= xyz[0, 1]
            for i, mid in enumerate(mocap_ids):
                data.mocap_pos[mid] = xyz[i]

            if retargeter is not None:
                try:
                    q_joints, _, _ = retargeter.retarget(pose)
                    for k, idx in enumerate(g1_dof_qpos_idx):
                        data.qpos[idx] = float(q_joints[k])
                    mujoco.mj_forward(model, data)
                except Exception as exc:
                    print(f"  retarget failed: {exc!r}", file=sys.stderr)

            viewer.sync()

            # Pace to the incoming rate when replaying an MCAP.
            if real_time and last_stamp_ns is not None:
                dt_ns = stamp_ns - last_stamp_ns
                if 0 < dt_ns < 1_000_000_000:  # cap sleep at 1s
                    time.sleep(dt_ns / 1e9)
            last_stamp_ns = stamp_ns


def main() -> int:
    _set_mujoco_gl_default()

    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mcap", type=Path, help="Replay this MCAP (file or rosbag2 dir).")
    src.add_argument("--live", action="store_true",
                     help="Subscribe to /pico_body_state via rclpy.")
    p.add_argument("--topic", default="/pico_body_state",
                   help="(--live) ROS2 topic name.")
    p.add_argument("--show", choices=["skeleton", "retarget", "wbt"],
                   default="skeleton",
                   help="What to render.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="(--mcap) Cap frames; 0 = full clip.")
    p.add_argument("--no-real-time", action="store_true",
                   help="(--mcap) Don't pace replay to MCAP timestamps.")
    args = p.parse_args()

    if args.mcap is not None:
        max_frames = args.max_frames if args.max_frames > 0 else None
        source = _mcap_source(args.mcap, max_frames)
        real_time = not args.no_real_time
    else:
        source = _live_source(args.topic)
        real_time = False  # already paced by the subscription

    _run_viewer(source, args.show, real_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
