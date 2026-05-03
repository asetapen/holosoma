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

Usage (host-side, all deps present in ~/lab42 Python):

    python3 holosoma_extensions/scripts/three_panel_eval.py \\
        --mcap holosoma_extensions/test_data/pico_example_long.mcap \\
        --onnx holosoma_extensions/models/active.onnx \\
        --out  /tmp/three_panel/three_panel.mp4

The --online mode (not implemented yet) will re-run through the full
ROS pipeline (pi run wbt-teleop --mujoco --bag) to isolate ROS-induced
error from pure-policy error.
"""

from __future__ import annotations

import argparse
import os
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

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
    from mcap.reader import make_reader

    poses = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, _, msg in reader.iter_messages(topics=["/pico_body_state"]):
            if len(msg.data) < 4 + 168 * 8:
                continue
            poses.append(_decode_pico_body_joint_poses(msg.data))
            if max_frames is not None and len(poses) >= max_frames:
                break
    import numpy as np

    return np.stack(poses, axis=0) if poses else np.zeros((0, 24, 7))


# ---------------------------------------------------------------------------
# Panel 1: pico stick-figure
# ---------------------------------------------------------------------------


def _render_pico_panel(pose_xyz, fig_canvas, ax, init_xyz):
    """Draw the 24-joint SMPL skeleton as a stick figure. Return RGB array."""
    import numpy as np

    ax.cla()

    # pico frame is Y-up; rotate to Z-up so it matches the MuJoCo panels.
    # (x, y, z) -> (x, -z, y) — same transform the retargeter uses.
    rotated = np.empty_like(pose_xyz)
    rotated[:, 0] = pose_xyz[:, 0]
    rotated[:, 1] = -pose_xyz[:, 2]
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
    ax.view_init(elev=15, azim=-135)
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
        # Pad/truncate defensively so a bad input doesn't crash the whole video.
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
    """Non-blocking, latest-wins source identical in contract to
    ``_DriverTrackingSource`` in holosoma_driver.worker.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, required=True,
                   help="Path to the /pico_body_state MCAP.")
    p.add_argument("--onnx", type=Path, required=True,
                   help="Path to the WBT ONNX policy (e.g. models/active.onnx).")
    p.add_argument("--preset", default="g1-29dof-wbt-dense",
                   help="Inference preset name. Must match the ONNX's training preset.")
    p.add_argument("--out", type=Path, default=Path("/tmp/three_panel/three_panel.mp4"))
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many pico frames (0 = full MCAP).")
    p.add_argument("--fps", type=int, default=25,
                   help="Output video fps. pico is ~50 Hz; 25 fps halves it.")
    p.add_argument("--online", action="store_true",
                   help="Run the full ROS pipeline (pi run wbt-teleop) to "
                   "produce panel 3 instead of the offline in-process policy. "
                   "NOT YET IMPLEMENTED.")
    args = p.parse_args()

    if args.online:
        print("error: --online mode is not yet implemented", file=sys.stderr)
        return 2

    if not args.mcap.is_file():
        print(f"error: mcap not found: {args.mcap}", file=sys.stderr)
        return 2
    if not args.onnx.is_file():
        print(f"error: onnx not found: {args.onnx}", file=sys.stderr)
        return 2

    import numpy as np

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load pico frames ──────────────────────────────────────────────
    print(f"Reading {args.mcap} ...", flush=True)
    max_frames = args.max_frames if args.max_frames > 0 else None
    poses = _load_pico_frames(args.mcap, max_frames)
    if poses.shape[0] == 0:
        print("no /pico_body_state frames", file=sys.stderr)
        return 2
    print(f"  loaded {poses.shape[0]} frames", flush=True)

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

    # ── Build panel 3 policy (WBT, offline) ───────────────────────────
    print(f"Constructing WBT policy (preset={args.preset}) ...", flush=True)
    tracking = _LocalTrackingSource()
    policy = _build_wbt_policy(str(args.onnx), args.preset, tracking)

    # robot_state_data shape: (1, 7 + num_dofs + 6 + num_dofs [+ 3])
    #   base_pos(3) + base_quat(4) + dof_pos + base_lin_vel(3) + base_ang_vel(3) + dof_vel
    num_dofs = policy.num_dofs
    base_obs_len = 7 + num_dofs + 6 + num_dofs
    robot_state = np.zeros((1, base_obs_len), dtype=np.float64)
    # upright base quat (wxyz)
    robot_state[0, 3] = 1.0
    # Prime dof_pos at the default angles so the policy's "dof_pos - default"
    # starts at zero on frame 0.
    robot_state[0, 7 : 7 + num_dofs] = policy.default_dof_angles

    # The policy's start() path wires ``use_policy_action = True`` only after
    # a StateCommand.START. Call the same helper the driver's autostart does.
    try:
        from holosoma_inference.inputs.api.commands import StateCommand
        policy._dispatch_command(StateCommand.START)
    except Exception as exc:
        print(f"warning: policy start dispatch failed: {exc}", file=sys.stderr)

    # ── Render loop ───────────────────────────────────────────────────
    import imageio.v3 as iio

    frames = []
    n_over_limit = 0
    print(f"Rendering {poses.shape[0]} frames ...", flush=True)
    for t in range(poses.shape[0]):
        pico_pose = poses[t]

        # Panel 1
        panel1 = _render_pico_panel(pico_pose[:, :3], canvas, ax3d, poses[0, :, :3])

        # Panel 2: retargeted G1 pose. Use the retargeter's full qpos —
        # its mink Configuration stores [x, y, z, qw, qx, qy, qz, joints]
        # (freejoint wxyz convention). We need the root orientation or the
        # rendered pose will look tilted (the IK leaves slack in the hips
        # that cancels out against the retargeter's chosen root rotation).
        try:
            q_ret, _, _ = retargeter.retarget(pico_pose)
            q_full = np.asarray(retargeter._config.q, dtype=np.float64)
            root_pos = q_full[:3].copy()
            root_quat_wxyz = q_full[3:7].copy()
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

        # Panel 3: WBT policy output
        payload = _build_tracking_payload(pico_pose)
        tracking.push(payload)
        # Update dof_pos in the fake robot_state to the *last* policy output
        # so observations evolve coherently frame-to-frame. On frame 0
        # this is default_q; on frame t > 0 it's the previous q_target.
        try:
            action = policy.rl_inference(robot_state)
            q_wbt = np.asarray(action[0]) + policy.default_dof_angles
            # Feed back: next frame's observations see the command we
            # just issued.
            robot_state[0, 7 : 7 + num_dofs] = q_wbt
        except Exception as exc:
            q_wbt = policy.default_dof_angles.copy()
            if t < 3:
                print(f"  [panel3] frame {t}: policy inference failed: {exc!r}",
                      file=sys.stderr)

        panel3 = _render_g1_pose(scene_wbt, q_wbt, "WBT policy output")

        composite = np.concatenate([panel1, panel2, panel3], axis=1)
        composite = _draw_frame_index(composite, t, poses.shape[0])
        frames.append(composite)

        if (t + 1) % 50 == 0 or t == poses.shape[0] - 1:
            print(f"  frame {t + 1}/{poses.shape[0]}", flush=True)

    # ── Write mp4 ─────────────────────────────────────────────────────
    print(f"Writing {args.out} ...", flush=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264")
    print(f"  wrote {args.out}")
    if n_over_limit:
        print(f"  over-limit retargeted frames: {n_over_limit}/{len(frames)}")

    scene_retarget.renderer.close()
    scene_wbt.renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
