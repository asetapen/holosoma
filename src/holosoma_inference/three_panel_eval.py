#!/usr/bin/env python3
"""Render a 3-panel motion-correctness evaluation video for a pico MCAP.

    [ Pico skeleton ] [ Retargeted G1 ] [ WBT policy output ]

Each frame of ``/pico_body_state`` in the MCAP becomes one video frame.
Panels are rendered independently, horizontally concatenated.

- Panel 1: stick-figure of the 24 SMPL body joints (matplotlib 3D).
- Panel 2: G1 29-DOF MJCF with q_joints = SMPLRetargeter(pico pose), applied
  kinematically (mj_forward — no physics).
- Panel 3: same G1 MJCF, but q_joints = WBT policy output. The policy is
  constructed with HOLOSOMA_ROBOT_BACKEND=mujoco and fed a
  TrackingPayload per frame via a local TrackingSource. We call
  ``policy.rl_inference`` directly and take ``scaled_policy_action +
  default_dof_angles`` as the per-frame q_target.

Usage (requires mujoco, mcap, matplotlib, imageio, pinocchio, and the
holosoma_inference + holosoma_retargeting packages on the Python path):

    python3 three_panel_eval.py \\
        --mcap path/to/pico_recording.mcap \\
        --onnx path/to/wbt_policy.onnx \\
        --out  /tmp/three_panel.mp4

The --online flag reads a pre-recorded rosbag2 MCAP containing
/holosoma_cmd and renders panel 3 from those timestamps instead of
running the policy in-process. Diffing offline vs online panel 3
isolates runtime-pipeline error from pure-policy error.
"""

from __future__ import annotations

import argparse
import os
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


def _set_mujoco_gl_default() -> None:
    """Default ``MUJOCO_GL`` to egl (linux) / glfw (darwin) for offscreen rendering.

    Called from ``main()`` so importing this module as a library does
    not mutate the caller's environment.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")


# SMPL 24-joint parent indices (body-only; standard SMPL kinematic tree).
# index = child, value = parent (-1 for root).
SMPL_PARENTS = [
    -1,  # 0 pelvis (root)
    0,   # 1 left_hip
    0,   # 2 right_hip
    0,   # 3 spine1
    1,   # 4 left_knee
    2,   # 5 right_knee
    3,   # 6 spine2
    4,   # 7 left_ankle
    5,   # 8 right_ankle
    6,   # 9 spine3
    7,   # 10 left_foot
    8,   # 11 right_foot
    9,   # 12 neck
    9,   # 13 left_collar
    9,   # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_hand
    21,  # 23 right_hand
]

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

PANEL_WIDTH = 480
PANEL_HEIGHT = 480


def _decode_pico_body_joint_poses(cdr_buf: bytes):
    """Deserialize the first 168 doubles (24 x 7) from a PicoBodyState CDR msg."""
    floats = struct.unpack_from("<168d", cdr_buf, 4)
    import numpy as np

    return np.asarray(floats, dtype=np.float64).reshape(24, 7)


def _load_pico_frames(mcap_path: Path, max_frames: int | None):
    """Load /pico_body_state frames from an MCAP file or rosbag2 dir.

    Returns (poses (T, 24, 7), stamps_ns (T,)). stamps use ``log_time``
    so the online-mode cross-reference to /holosoma_cmd works on the
    same clock.
    """
    from mcap.reader import make_reader
    import numpy as np

    mcaps: list[Path]
    if Path(mcap_path).is_dir():
        mcaps = sorted(Path(mcap_path).glob("*.mcap"))
        if not mcaps:
            raise FileNotFoundError(f"no .mcap files under {mcap_path}")
    else:
        mcaps = [Path(mcap_path)]

    poses = []
    times_ns = []
    for mp in mcaps:
        with open(mp, "rb") as f:
            reader = make_reader(f)
            for _, _, msg in reader.iter_messages(topics=["/pico_body_state"]):
                if len(msg.data) < 4 + 168 * 8:
                    continue
                poses.append(_decode_pico_body_joint_poses(msg.data))
                times_ns.append(msg.log_time)
                if max_frames is not None and len(poses) >= max_frames:
                    break
        if max_frames is not None and len(poses) >= max_frames:
            break

    if not poses:
        return np.zeros((0, 24, 7)), np.zeros((0,), dtype=np.int64)

    # Sanity-check the first frame: the pelvis quaternion (cols 3-7)
    # must have unit norm. If the schema ever grows a field before
    # body_joint_poses, the hardcoded CDR offset of 4 lands us at the
    # wrong floats and this check catches it loudly before the policy
    # sees garbage.
    pelvis_quat = poses[0][0, 3:7]
    quat_norm = float(np.linalg.norm(pelvis_quat))
    if not 0.9 < quat_norm < 1.1:
        raise RuntimeError(
            f"first-frame pelvis quaternion has norm {quat_norm:.3f}, expected ~1.0; "
            "the /pico_body_state CDR layout may have changed — decoder assumes "
            "body_joint_poses is the first field."
        )
    return np.stack(poses, axis=0), np.asarray(times_ns, dtype=np.int64)


# ---------------------------------------------------------------------------
# Online-mode rosbag parser
# ---------------------------------------------------------------------------


def _cdr_align(buf: bytes, off: int, boundary: int) -> int:
    """CDR padding: round ``off`` up to the next ``boundary``. The encoding
    header is 4 bytes; alignment inside the payload is relative to (off - 4).
    """
    payload_off = off - 4
    pad = (-payload_off) % boundary
    return off + pad


def _cdr_read_string(buf: bytes, off: int) -> tuple[str, int]:
    off = _cdr_align(buf, off, 4)
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    # n includes the trailing NUL
    s = buf[off : off + n - 1].decode("utf-8", errors="replace")
    off += n
    return s, off


def _decode_joint_trajectory_cdr(cdr_buf: bytes):
    """Decode a JointTrajectory CDR payload.

    Layout (post 4-byte encapsulation header):
        Header
            int32  sec            # builtin_interfaces/Time stores sec
            uint32 nanosec        # as (sec: int32, nanosec: uint32)
            string frame_id
        string[] joint_names
        float32[] positions
        float32[] velocities
        float32[] accelerations
    Returns (stamp_ns, joint_names, positions, velocities, accelerations).
    """
    import numpy as np

    off = 4  # skip encapsulation header
    off = _cdr_align(cdr_buf, off, 4)
    # sec is int32 (signed) per builtin_interfaces/Time; unpack accordingly
    # so negative time stamps (rare, but possible on clock drift) don't
    # become huge unsigned values.
    (sec,) = struct.unpack_from("<i", cdr_buf, off); off += 4
    (nanosec,) = struct.unpack_from("<I", cdr_buf, off); off += 4
    stamp_ns = int(sec) * 1_000_000_000 + int(nanosec)
    _frame_id, off = _cdr_read_string(cdr_buf, off)

    # string[] joint_names
    off = _cdr_align(cdr_buf, off, 4)
    (n_names,) = struct.unpack_from("<I", cdr_buf, off); off += 4
    names = []
    for _ in range(n_names):
        s, off = _cdr_read_string(cdr_buf, off)
        names.append(s)

    # float32[] positions
    off = _cdr_align(cdr_buf, off, 4)
    (n_pos,) = struct.unpack_from("<I", cdr_buf, off); off += 4
    off = _cdr_align(cdr_buf, off, 4)
    positions = np.frombuffer(cdr_buf, dtype=np.float32, count=n_pos, offset=off).copy()
    off += 4 * n_pos

    # float32[] velocities
    off = _cdr_align(cdr_buf, off, 4)
    (n_vel,) = struct.unpack_from("<I", cdr_buf, off); off += 4
    off = _cdr_align(cdr_buf, off, 4)
    velocities = np.frombuffer(cdr_buf, dtype=np.float32, count=n_vel, offset=off).copy()
    off += 4 * n_vel

    # float32[] accelerations
    off = _cdr_align(cdr_buf, off, 4)
    (n_acc,) = struct.unpack_from("<I", cdr_buf, off); off += 4
    off = _cdr_align(cdr_buf, off, 4)
    accelerations = np.frombuffer(cdr_buf, dtype=np.float32, count=n_acc, offset=off).copy()

    return stamp_ns, names, positions, velocities, accelerations


def _load_holosoma_cmd_bag(bag_path: Path):
    """Read a rosbag2 MCAP and return sorted (stamps_ns, q_targets) arrays.

    Handles both ``bag.mcap`` direct path and rosbag2 directories (scans
    for *.mcap inside).
    """
    from mcap.reader import make_reader
    import numpy as np

    mcaps: list[Path]
    if bag_path.is_dir():
        mcaps = sorted(bag_path.glob("*.mcap"))
        if not mcaps:
            raise FileNotFoundError(f"no .mcap files under {bag_path}")
    else:
        mcaps = [bag_path]

    stamps = []
    q_targets = []
    n_decode_errors = 0
    first_decode_err: Exception | None = None
    for mp in mcaps:
        with open(mp, "rb") as f:
            reader = make_reader(f)
            for _, _, msg in reader.iter_messages(topics=["/holosoma_cmd"]):
                try:
                    stamp_ns, _names, pos, _v, _a = _decode_joint_trajectory_cdr(msg.data)
                except (struct.error, UnicodeDecodeError, ValueError, EOFError) as exc:
                    # Bags sometimes have a malformed first/last message;
                    # skip rather than fail the whole run, but count
                    # failures and surface the first one so a schema
                    # change doesn't silently drop every frame.
                    # Narrow exception type: we want KeyboardInterrupt /
                    # MemoryError / SystemExit to still propagate.
                    # See 2026-05-05 code review #14.
                    n_decode_errors += 1
                    if first_decode_err is None:
                        first_decode_err = exc
                        print(f"  [holosoma_cmd decode] first failure: {exc!r}",
                              file=sys.stderr)
                    continue
                if pos.size == 0:
                    continue
                # Prefer stamp from header; fall back to log_time.
                if stamp_ns <= 0:
                    stamp_ns = int(msg.log_time)
                stamps.append(stamp_ns)
                q_targets.append(pos.astype(np.float64))
    if not stamps:
        raise RuntimeError(f"no /holosoma_cmd messages in {bag_path}")
    if n_decode_errors:
        print(f"  [holosoma_cmd decode] {n_decode_errors} messages skipped "
              f"(first error above)", file=sys.stderr)
    stamps = np.asarray(stamps, dtype=np.int64)
    # Some cmds may differ in length across frames (shouldn't, but guard).
    dof = max(q.shape[0] for q in q_targets)
    q_arr = np.zeros((len(q_targets), dof), dtype=np.float64)
    for i, q in enumerate(q_targets):
        q_arr[i, : q.shape[0]] = q
    order = np.argsort(stamps)
    return stamps[order], q_arr[order]


def _nearest_stamp_idx(sorted_stamps, target_ns):
    """Find index in ``sorted_stamps`` closest to ``target_ns``."""
    import numpy as np

    i = int(np.searchsorted(sorted_stamps, target_ns))
    if i == 0:
        return 0
    if i >= len(sorted_stamps):
        return len(sorted_stamps) - 1
    if abs(sorted_stamps[i - 1] - target_ns) <= abs(sorted_stamps[i] - target_ns):
        return i - 1
    return i


# ---------------------------------------------------------------------------
# Panel 1: pico stick-figure
# ---------------------------------------------------------------------------


def _render_pico_panel(pose_xyz, fig_canvas, ax):
    """Draw the 24-joint SMPL skeleton as a stick figure. Return RGB array."""
    import numpy as np

    ax.cla()

    # pico frame is Y-up; rotate to Z-up, then apply a 90° CCW yaw
    # (viewed from above) so the skeleton faces the same direction as
    # the G1 in the MuJoCo panels.
    # Z-up:  (x, y, z) -> (x, -z, y)
    # +90 CCW yaw around Z:  (x, y, z) -> (-y, x, z)
    # Composed: (x, y, z) -> (z, x, y)
    rotated = np.empty_like(pose_xyz)
    rotated[:, 0] = pose_xyz[:, 2]
    rotated[:, 1] = pose_xyz[:, 0]
    rotated[:, 2] = pose_xyz[:, 1]

    # Pin the pelvis at (0, 0) horizontally so the figure doesn't drift off
    # the camera — leave z (height) unchanged so stepping/squatting is
    # visible.
    rotated[:, 0] -= rotated[0, 0]
    rotated[:, 1] -= rotated[0, 1]

    for child, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        ax.plot(
            [rotated[parent, 0], rotated[child, 0]],
            [rotated[parent, 1], rotated[child, 1]],
            [rotated[parent, 2], rotated[child, 2]],
            color="#66ccff", linewidth=2,
        )
    ax.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c="#ffaa66", s=12)

    # Fixed axes so the figure doesn't jitter with per-frame extent.
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.9, 1.0)
    ax.set_box_aspect((1, 1, 1.6))
    ax.set_title("Pico skeleton (SMPL-24)", color="white")
    ax.set_facecolor("black")
    # Match the MuJoCo panel cameras (azimuth=135, elevation=-15 in
    # mujoco convention). matplotlib's ``elev`` is the angle above the
    # XY plane of the camera position (positive = looking down);
    # mujoco's ``elevation`` is the pitch of the look-direction
    # (negative = looking down). So a mujoco elevation of -15 pairs
    # with a matplotlib elev of +15. Azimuth has the same convention
    # in both (degrees CCW around +Z from +X).
    ax.view_init(elev=15, azim=135)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#222222")

    fig_canvas.draw()
    w, h = fig_canvas.get_width_height()
    buf = np.frombuffer(fig_canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


# ---------------------------------------------------------------------------
# Panels 2 + 3: MuJoCo offscreen rendering of a G1 pose.
# ---------------------------------------------------------------------------


@dataclass
class G1Scene:
    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    renderer: "mujoco.Renderer"
    camera: "mujoco.MjvCamera"
    n_dof: int
    default_q: "np.ndarray"


def _build_g1_scene() -> G1Scene:
    import mujoco
    import numpy as np
    import holosoma_retargeting as _hr

    xml = Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 135.0
    cam.elevation = -15.0
    cam.distance = 3.4
    cam.lookat[:] = [0.0, 0.0, 0.9]

    from holosoma_inference.config.config_values.robot import g1_29dof as g1_cfg

    default_q = np.array(g1_cfg.default_dof_angles, dtype=np.float64)
    return G1Scene(model=model, data=data, renderer=renderer, camera=cam,
                   n_dof=29, default_q=default_q)


_render_g1_pose_warned_dof_mismatch = False


def _render_g1_pose(scene: G1Scene, q_joints, label: str,
                    root_pos=None, root_quat_wxyz=None) -> "np.ndarray":
    """Set qpos[7:36] = q_joints and (optionally) freejoint pose, mj_forward, render.

    ``root_pos`` is (3,) in world coords; ``root_quat_wxyz`` is (4,). If
    omitted, we pin the pelvis at a standing height with identity
    orientation. Use the retargeter's computed root for panel 2 so the
    rendered pose matches the skeleton orientation.
    """
    import mujoco
    import numpy as np

    q = np.asarray(q_joints, dtype=np.float64).reshape(-1)
    if q.shape[0] != scene.n_dof:
        # Pad/truncate defensively so a bad input doesn't crash the whole
        # video, but warn the first time so a configuration drift (e.g.
        # a 25-DOF checkpoint feeding a 29-DOF scene) isn't silent.
        global _render_g1_pose_warned_dof_mismatch
        if not _render_g1_pose_warned_dof_mismatch:
            print(f"  warning: q_joints has {q.shape[0]} DOF, scene expects "
                  f"{scene.n_dof}: padding/truncating (further warnings suppressed).",
                  file=sys.stderr)
            _render_g1_pose_warned_dof_mismatch = True
        fixed = np.zeros(scene.n_dof)
        fixed[: min(scene.n_dof, q.shape[0])] = q[: min(scene.n_dof, q.shape[0])]
        q = fixed
    if root_pos is None:
        scene.data.qpos[0:3] = [0.0, 0.0, 0.9]
    else:
        scene.data.qpos[0:3] = root_pos
    if root_quat_wxyz is None:
        scene.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    else:
        scene.data.qpos[3:7] = root_quat_wxyz
    scene.data.qpos[7: 7 + scene.n_dof] = q
    scene.data.qvel[:] = 0.0
    mujoco.mj_forward(scene.model, scene.data)
    scene.renderer.update_scene(scene.data, camera=scene.camera)
    frame = scene.renderer.render().copy()
    return _draw_label(frame, label)


def _draw_label(img, text: str):
    """Overlay a short label at the top of a panel."""
    import numpy as np

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return img

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([(0, 0), (img.shape[1], 28)], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def _draw_frame_index(img, t: int, total: int):
    """Burn the frame index into the bottom-right so viewers can jump."""
    import numpy as np

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return img

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    text = f"frame {t + 1}/{total}"
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = len(text) * 9, 16
    x = img.shape[1] - w - 10
    y = img.shape[0] - h - 10
    d.rectangle([(x - 4, y - 3), (x + w + 4, y + h + 6)], fill=(0, 0, 0))
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


# ---------------------------------------------------------------------------
# WBT policy harness (offline)
# ---------------------------------------------------------------------------


class _LocalTrackingSource:
    """Non-blocking, latest-wins ``TrackingSource`` implementation.

    Conforms to the protocol defined in
    ``holosoma_inference.policies.tracking_source.TrackingSource``:
    ``get_latest()`` is non-blocking and returns the most recent
    pushed payload (or ``None`` if nothing new has arrived since the
    previous call).
    """

    def __init__(self):
        self._latest = None

    def push(self, payload):
        self._latest = payload

    def get_latest(self):
        p, self._latest = self._latest, None
        return p


def _build_wbt_policy(onnx_path: str, preset: str, source: _LocalTrackingSource):
    """Construct a WholeBodyTrackingPolicy that writes to a MuJoCo interface
    (so create_interface doesn't block on a real SDK) and takes
    TrackingPayloads from a local source.
    """
    os.environ["HOLOSOMA_ROBOT_BACKEND"] = "mujoco"
    os.environ["HOLOSOMA_MUJOCO_REAL_TIME"] = "0"

    import dataclasses as _dc
    from holosoma_inference.config.config_values.inference import get_defaults
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

    cfg_defaults = get_defaults()
    inf_cfg = cfg_defaults[preset]
    inf_cfg = _dc.replace(inf_cfg, task=_dc.replace(inf_cfg.task, model_path=onnx_path))

    # The interface has to resolve to something that doesn't touch real HW.
    inf_cfg = _dc.replace(inf_cfg, task=_dc.replace(inf_cfg.task, interface="lo"))

    # WBT's _retarget_payload_to_motion_command needs a URDF to construct
    # its own SMPLRetargeter; without it the policy logs WARN and falls
    # through to its bundled ONNX-clip motion_command — we'd be rendering
    # the default dance clip instead of the pico teleop input. Point it
    # at the shipped G1 MJCF.
    import holosoma_retargeting as _hr

    mjcf = str(Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml")
    if not inf_cfg.robot.urdf_path:
        inf_cfg = _dc.replace(inf_cfg, robot=_dc.replace(inf_cfg.robot, urdf_path=mjcf))

    policy = WholeBodyTrackingPolicy(inf_cfg, tracking_source=source)
    return policy


def _build_tracking_payload(pose_xyz_quat):
    """Build a TrackingPayload carrying the SMPL 24-joint transforms,
    matching _RealHolosomaDriver._build_tracking_payload.
    """
    import numpy as np
    from holosoma_inference.policies.tracking_source import TrackingPayload

    # joint_transforms: (N,7) flattened per-joint (x, y, z, qx, qy, qz, qw)
    flat = np.asarray(pose_xyz_quat, dtype=np.float32).reshape(-1)
    return TrackingPayload(
        joint_names=list(SMPL_JOINT_NAMES),
        joint_transforms=flat,
        joint_confidences=np.ones(24, dtype=np.float32),
        device_type="pico_replay",
        tracking_quality=1,
        mode=2,  # INFERENCE
        forward_gripper_commands=False,
    )


# ---------------------------------------------------------------------------
# Main composition loop
# ---------------------------------------------------------------------------


def main() -> int:
    _set_mujoco_gl_default()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, default=None,
                   help="Path to the /pico_body_state MCAP. Required "
                   "unless --online is set (in which case pico frames "
                   "come from the online bag).")
    p.add_argument("--onnx", type=Path, required=True,
                   help="Path to the WBT ONNX policy (e.g. models/active.onnx).")
    p.add_argument("--preset", default="g1-29dof-wbt-dense",
                   help="Inference preset name. Must match the ONNX's training preset.")
    p.add_argument("--out", type=Path, default=Path("/tmp/three_panel/three_panel.mp4"))
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many pico frames (0 = full MCAP).")
    p.add_argument("--fps", type=int, default=25,
                   help="Output video fps. pico is ~50 Hz; 25 fps halves it.")
    p.add_argument("--online", type=Path, default=None,
                   help="Path to a rosbag2 MCAP containing both "
                   "/pico_body_state (panels 1+2 source) and "
                   "/holosoma_cmd (panel 3) captured from a live "
                   "deployment run. When set, --mcap is ignored and "
                   "the online bag is the sole source for all three "
                   "panels. Diffing the offline (no --online) vs "
                   "online panel 3 at matched pico timestamps isolates "
                   "runtime-pipeline error from pure policy error.")
    p.add_argument("--holosoma-reference", type=Path, default=None,
                   help="Path to an NPZ produced by running the policy "
                   "through its own headless evaluation path with "
                   "HOLOSOMA_DEBUG_LOG set (the debug log contains per-"
                   "frame ``action`` and ``dof_pos``). When set, a "
                   "4th panel appears showing action + default_dof "
                   "from the reference run for each frame, so you can "
                   "diff pure-holosoma output against our in-process "
                   "WBT output and the ROS-bag /holosoma_cmd.")
    args = p.parse_args()

    online_bag: Path | None = args.online

    if online_bag is None:
        if args.mcap is None:
            print("error: --mcap is required unless --online is set", file=sys.stderr)
            return 2
        if not args.mcap.is_file():
            print(f"error: mcap not found: {args.mcap}", file=sys.stderr)
            return 2
        if not args.onnx.is_file():
            print(f"error: onnx not found: {args.onnx}", file=sys.stderr)
            return 2
    else:
        if not (online_bag.is_file() or online_bag.is_dir()):
            print(f"error: online bag not found: {online_bag}", file=sys.stderr)
            return 2

    import numpy as np

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load pico frames ──────────────────────────────────────────────
    # In --online mode, pico frames come from the online bag itself so the
    # timestamps align with its /holosoma_cmd stream.
    pico_source = online_bag if online_bag is not None else args.mcap
    print(f"Reading pico frames from {pico_source} ...", flush=True)
    max_frames = args.max_frames if args.max_frames > 0 else None
    poses, pico_stamps = _load_pico_frames(pico_source, max_frames)
    if poses.shape[0] == 0:
        print("no /pico_body_state frames", file=sys.stderr)
        return 2
    print(f"  loaded {poses.shape[0]} frames", flush=True)

    online_stamps = None
    online_q = None
    if online_bag is not None:
        print(f"Reading /holosoma_cmd from {online_bag} ...", flush=True)
        online_stamps, online_q = _load_holosoma_cmd_bag(online_bag)
        if online_q.shape[1] < 29:
            print(f"error: /holosoma_cmd messages only carry {online_q.shape[1]} DOF; "
                  "expected at least 29 for G1", file=sys.stderr)
            return 2
        if online_q.shape[1] > 29:
            # Plausible future schema — gripper slots after the arm DOFs.
            # The renderer pins pelvis to identity so the extra DOFs are
            # silently dropped; warn so this isn't hidden.
            print(f"warning: /holosoma_cmd carries {online_q.shape[1]} DOF; "
                  "panel 3 renders the first 29 only (extras ignored).",
                  file=sys.stderr)
        print(f"  loaded {online_stamps.shape[0]} /holosoma_cmd messages "
              f"(dof={online_q.shape[1]})", flush=True)

    # Load the pure-holosoma reference debug NPZ if provided. Prefer
    # ``dof_pos`` (the actual post-PD sim state) over ``action``: the raw
    # action values are ONNX-space deltas that get PD-smoothed by the
    # MuJoCo interface, so rendering ``action + default_dof`` directly
    # produces kinematically-extreme poses that don't reflect what the
    # policy actually achieved. ``dof_pos`` is the ground-truth
    # trajectory the policy executed.
    ref_dof_pos = None
    if args.holosoma_reference is not None:
        print(f"Reading holosoma reference debug {args.holosoma_reference} ...",
              flush=True)
        ref_d = np.load(args.holosoma_reference)
        if "dof_pos" in ref_d.files:
            ref_dof_pos = np.asarray(ref_d["dof_pos"], dtype=np.float64)
            print(f"  {ref_dof_pos.shape[0]} reference dof_pos frames, "
                  f"DOF={ref_dof_pos.shape[1]}", flush=True)
        else:
            # Fallback for older debug NPZs without dof_pos (action + default
            # gives the commanded set-point, not the achieved state — less
            # faithful but still informative).
            ref_dof_pos = np.asarray(ref_d["action"], dtype=np.float64)
            print(f"  WARN: debug NPZ has no 'dof_pos'; falling back to "
                  f"'action' (commanded set-point, not achieved state).",
                  file=sys.stderr)

    # ── Build panel 1 figure ──────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(PANEL_WIDTH / 100, PANEL_HEIGHT / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax3d = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")

    # ── Build panel 2 scene (retargeted) ──────────────────────────────
    print("Building retargeter + G1 scenes ...", flush=True)
    from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter
    import holosoma_retargeting as _hr

    mjcf = Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"
    retargeter = SMPLRetargeter(str(mjcf), max_ik_iters=4)
    scene_retarget = _build_g1_scene()
    scene_wbt = _build_g1_scene()
    scene_ref = _build_g1_scene() if ref_dof_pos is not None else None

    # ── Build panel 3 policy (WBT, offline) — skipped in --online mode ─
    policy = None
    tracking = None
    robot_state = None
    num_dofs = 29
    if online_bag is None:
        print(f"Constructing WBT policy (preset={args.preset}) ...", flush=True)
        tracking = _LocalTrackingSource()
        policy = _build_wbt_policy(str(args.onnx), args.preset, tracking)

        # robot_state_data shape: (1, 7 + num_dofs + 6 + num_dofs [+ 3])
        # base_pos(3) + base_quat(4) + dof_pos + base_lin_vel(3) + base_ang_vel(3) + dof_vel
        num_dofs = policy.num_dofs
        base_obs_len = 7 + num_dofs + 6 + num_dofs
        robot_state = np.zeros((1, base_obs_len), dtype=np.float64)
        # upright base quat (wxyz)
        robot_state[0, 3] = 1.0
        # Prime dof_pos at the default angles so the policy's "dof_pos - default"
        # starts at zero on frame 0.
        robot_state[0, 7 : 7 + num_dofs] = policy.default_dof_angles

        # The policy's start() path wires ``use_policy_action = True`` only
        # after a StateCommand.START. Call the same helper the driver's
        # autostart does. Hard fail on failure (2026-05-05 code review
        # #10): three_panel_eval is a diagnostic harness whose entire point
        # is comparing panel-3 policy output to panels 1/2. Silently
        # swallowing the START dispatch and running with
        # ``use_policy_action = False`` would render panel 3 at the default
        # pose and make the diagnostic misleading.
        from holosoma_inference.inputs.api.commands import StateCommand
        policy._dispatch_command(StateCommand.START)
    else:
        print("Skipping in-process WBT policy construction (--online mode).", flush=True)

    # ── Render loop ───────────────────────────────────────────────────
    import imageio.v3 as iio

    frames = []
    print(f"Rendering {poses.shape[0]} frames ...", flush=True)
    # One-shot vertical lift for panel 2 — see Panel-2 block below.
    z_lift: float | None = None
    for t in range(poses.shape[0]):
        pico_pose = poses[t]

        # Panel 1
        panel1 = _render_pico_panel(pico_pose[:, :3], canvas, ax3d)

        # Panel 2: retargeted G1 pose. Use the retargeter's full qpos —
        # its mink Configuration stores [x, y, z, qw, qx, qy, qz, joints]
        # (freejoint wxyz convention). We need the root orientation or the
        # rendered pose will look tilted (the IK leaves slack in the hips
        # that cancels out against the retargeter's chosen root rotation).
        try:
            q_ret, _, _ = retargeter.retarget(pico_pose)
            root_pos = retargeter.last_root_pos.copy()
            root_quat_wxyz = retargeter.last_root_quat_wxyz.copy()
            # The retargeter runs ``_anchor_to_ground`` which drops the
            # skeleton so feet sit at 5 cm — which puts the *pelvis*
            # near the floor and makes the rendered robot look like
            # it's squatting in a hole. Lift every frame by the same
            # one-shot offset so frame 0's pelvis sits at the neutral
            # standing height (retargeter._g1_leg, measured via FK).
            # Relative-to-pelvis motion (squats, steps, leans) is
            # preserved because the shift is constant.
            if z_lift is None:
                z_lift = float(retargeter._g1_leg) - float(root_pos[2])
            root_pos[2] += z_lift
            # Planar-anchor: center the robot so all frames sit in the
            # viewport. Preserve height so squats/stands are visible.
            root_pos[0] = 0.0
            root_pos[1] = 0.0
        except Exception as exc:
            q_ret = scene_retarget.default_q.copy()
            root_pos = None
            root_quat_wxyz = None
            if t < 3:
                print(f"  [panel2] frame {t}: retarget failed: {exc!r}",
                      file=sys.stderr)
        panel2 = _render_g1_pose(scene_retarget, q_ret,
                                 "Retargeted (G1 29-DOF)",
                                 root_pos=root_pos,
                                 root_quat_wxyz=root_quat_wxyz)

        # Panel 3: WBT output. Sources, in priority order:
        #   (a) --online bag  → nearest /holosoma_cmd message
        #   (b) --holosoma-reference debug NPZ → dof_pos (achieved sim state)
        #   (c) in-process WBT policy with fake robot_state (DEPRECATED —
        #       kept as last-resort fallback; feeds the policy a synthetic
        #       always-upright base state so output is unphysical. Use
        #       only for sanity checks.)
        if online_bag is not None:
            idx = _nearest_stamp_idx(online_stamps, int(pico_stamps[t]))
            q_wbt = online_q[idx, :29]
            panel3_label = f"WBT /holosoma_cmd (online bag, +{(online_stamps[idx] - pico_stamps[t]) / 1e6:+.0f}ms)"
        elif ref_dof_pos is not None:
            # Prefer the real headless-sim trajectory from the reference
            # debug NPZ — same data as panel 4 but this makes panel 3
            # meaningful when the in-process fake-obs path was broken.
            ref_idx = min(t, ref_dof_pos.shape[0] - 1)
            q_wbt = ref_dof_pos[ref_idx]
            panel3_label = f"WBT headless-sim dof_pos[{ref_idx}]"
        else:
            payload = _build_tracking_payload(pico_pose)
            tracking.push(payload)
            try:
                action = policy.rl_inference(robot_state)
                q_wbt = np.asarray(action[0]) + policy.default_dof_angles
                robot_state[0, 7 : 7 + num_dofs] = q_wbt
            except Exception as exc:
                q_wbt = policy.default_dof_angles.copy()
                if t < 3:
                    print(f"  [panel3] frame {t}: policy inference failed: {exc!r}",
                          file=sys.stderr)
            panel3_label = "WBT policy output (offline, kinematic feedback)"

        panel3 = _render_g1_pose(scene_wbt, q_wbt, panel3_label)

        panels = [panel1, panel2, panel3]
        if scene_ref is not None:
            # Render the holosoma-reference trajectory directly from the
            # debug NPZ's ``dof_pos`` — this is the actual post-PD sim
            # state at each frame, which is what the policy executed on
            # the MuJoCo interface. (Rendering ``action * scale + default``
            # produces the commanded set-point before PD smoothing, which
            # is kinematically extreme for frames where the raw action is
            # large and does not match what dense.mp4 showed.)
            ref_idx = min(t, ref_dof_pos.shape[0] - 1)
            q_ref = ref_dof_pos[ref_idx]
            panels.append(_render_g1_pose(
                scene_ref, q_ref, f"Holosoma ref (dof_pos[{ref_idx}])"))

        composite = np.concatenate(panels, axis=1)
        composite = _draw_frame_index(composite, t, poses.shape[0])
        frames.append(composite)

        if (t + 1) % 50 == 0 or t == poses.shape[0] - 1:
            print(f"  frame {t + 1}/{poses.shape[0]}", flush=True)

    # ── Write mp4 ─────────────────────────────────────────────────────
    print(f"Writing {args.out} ...", flush=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264")
    print(f"  wrote {args.out}")

    scene_retarget.renderer.close()
    scene_wbt.renderer.close()
    if scene_ref is not None:
        scene_ref.renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
