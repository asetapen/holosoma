#!/usr/bin/env python3
"""Real-time viewer of (q_target, dof_pos) from the WBT policy.

Renders two G1 URDFs side-by-side inside a single mujoco.viewer window:

  * Left  (x = -0.8 m, blue tint)  — driven by ``q_target`` (what the
                                     ONNX is commanding the motors to do).
  * Right (x = +0.8 m, orange tint) — driven by ``dof_pos`` (what the
                                     robot/sim actually reports back).

Data source: shared memory segment ``holosoma_policy_output``, written
per policy tick by ``teleop_wbt.py``. Also writes a running log of the
largest per-joint command-minus-actual diff so you can spot a joint
index mismatch (e.g. the policy commanding an elbow but the robot
responding with a knee) from the console while watching the 3D view.

Usage (inside sim/robot docker container while `pi run wbt-teleop ...`
is already running):

    bazel run //holosoma_extensions/thirdparty/holosoma/src/holosoma_inference:policy_output_viewer

    # With a different SHM name (for two concurrent sessions):
    bazel run //...:policy_output_viewer -- --shm-name my_policy_output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _set_mujoco_gl_default() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")


def _find_g1_mjcf() -> Path:
    import holosoma_retargeting as _hr

    return Path(_hr.__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"


def _build_dual_g1_mjcf(g1_xml_path: Path, offset: float = 0.8) -> str:
    """Produce an MJCF with two copies of the G1, one at x=-offset (command)
    and one at x=+offset (actual). Joint names are suffixed _cmd / _act so
    MuJoCo accepts the duplicates."""
    import re

    raw = g1_xml_path.read_text(encoding="utf-8")

    # We need to rename every named element (body / joint / geom / site / ...)
    # so the two copies don't collide. The dumb-but-reliable approach: a
    # regex on name="..." / childclass="..." / ... etc. is fragile. Simpler
    # approach: use MuJoCo's attach-at-worldbody mechanism via <include> is
    # also cross-file. So we do the rename inline — only name, joint=, body=
    # refs — which covers the G1 MJCF shape.
    def suffix_names(xml: str, suffix: str) -> str:
        # Attribute refs we need to keep consistent. Order matters: do
        # `meshname`-like bare refs AFTER `name=` so we don't double-suffix.
        attrs = ["name", "joint", "body1", "body2", "site1", "site2", "target", "objname"]
        out = xml
        for a in attrs:
            out = re.sub(rf'{a}="([^"]+)"', lambda m, a=a: f'{a}="{m.group(1)}{suffix}"', out)
        # Name the freejoint so we can address its qpos via mj_name2id later.
        # MJCF allows bare `<freejoint/>`; MuJoCo gives those an empty name,
        # so suffix_names above doesn't catch them.
        out = out.replace("<freejoint/>", f'<freejoint name="root{suffix}"/>')
        return out

    # Strip the ground plane / worldbody lights from the per-copy template so
    # the two copies don't spawn duplicate planes/lights. We keep one shared
    # copy in the combined <worldbody>.
    m_open = raw.find("<worldbody>")
    m_close = raw.find("</worldbody>")
    if m_open < 0 or m_close < 0:
        raise RuntimeError(f"G1 MJCF {g1_xml_path} missing <worldbody>")
    prefix = raw[:m_open]
    worldbody_inner = raw[m_open + len("<worldbody>") : m_close]
    trailer = raw[m_close + len("</worldbody>") :]

    # Drop any top-level <geom type="plane" .../> and <light .../> from the
    # per-copy worldbody so the combined scene has only one of each.
    template_for_copies = re.sub(r'<geom\s+type="plane"[^/]*/>', "", worldbody_inner)
    template_for_copies = re.sub(r"<light[^/]*/>", "", template_for_copies)

    cmd = suffix_names(template_for_copies, "_cmd")
    act = suffix_names(template_for_copies, "_act")

    # The first <body ...> in each copy is the pelvis (root). Inject a
    # pos= attribute on it so the freejoint stays a direct worldbody child
    # (MuJoCo requires this). pos is the freejoint's reference frame; with
    # qpos at identity the pelvis sits at (±offset, 0, 0.793).
    def set_root_pos(xml: str, x: float) -> str:
        return re.sub(
            r"<body(\s[^>]*?name=\"pelvis[^\"]*\"[^>]*)>",
            lambda m: f'<body{m.group(1)} pos="{x} 0 0.793">',
            xml,
            count=1,
        )

    cmd_positioned = set_root_pos(cmd, -offset)
    act_positioned = set_root_pos(act, +offset)

    # Shared ground + light + an explicit camera framing both copies.
    shared_scene = (
        '<geom type="plane" size="4 4 0.1" rgba="0.2 0.3 0.4 1"/>'
        '<light pos="0 0 4" castshadow="true"/>'
        '<camera name="dual_view" pos="0 -3.0 1.5" xyaxes="1 0 0 0 0.5 0.866" mode="fixed"/>'
    )
    dual_worldbody = f"<worldbody>\n{shared_scene}\n{cmd_positioned}\n{act_positioned}\n</worldbody>"
    return prefix + dual_worldbody + trailer


def _resolve_dof_qpos_addrs(model, suffix: str) -> tuple[list[int], list[str]]:
    """Return qpos addresses + bare joint names (minus suffix) for the 29
    actuated joints of one G1 copy.

    Enumerates MuJoCo joints directly (no retargeter dep), filtering out
    the freejoint. Names come from the combined scene with the suffix
    stripped so the caller can keep them aligned with the writer's DOF
    ordering (which is defined by the G1 MJCF joint list — same source).
    """
    import mujoco

    qpos_addrs: list[int] = []
    dof_names: list[str] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(j).name
        if not name.endswith(suffix):
            continue
        qpos_addrs.append(int(model.jnt_qposadr[j]))
        dof_names.append(name[: -len(suffix)])
    if not qpos_addrs:
        raise RuntimeError(f"combined scene has no joints with suffix {suffix!r}")
    return qpos_addrs, dof_names


def _resolve_freejoint_qpos(model, suffix: str) -> int | None:
    """Return the qpos address of the freejoint for one copy, or None."""
    import mujoco

    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            name = model.joint(j).name
            if name.endswith(suffix):
                return int(model.jnt_qposadr[j])
    return None


class _PolicyOutputShmReader:
    """Inlined reader for the `holosoma_policy_output` SHM segment.

    Kept in-file so the bazel target depends only on mujoco+numpy (no
    holosoma_ext_viser dep, which isn't available inside bazel-ci).
    Layout must stay in sync with PolicyOutputShmWriter in
    holosoma_ext_viser/skeleton_shm.py.
    """

    _HEADER_SIZE = 8

    def __init__(self, num_dofs: int = 29, shm_name: str = "holosoma_policy_output") -> None:
        from multiprocessing import shared_memory

        self._shared_memory = shared_memory
        self._shm_name = shm_name
        self._num_dofs = num_dofs
        self._joint_size = num_dofs * 8
        self._shm = None
        self._q_target = None
        self._dof_pos = None
        self._last_timestamp = 0.0

    def _try_connect(self) -> bool:
        import numpy as np

        if self._shm is not None:
            return True
        try:
            self._shm = self._shared_memory.SharedMemory(name=self._shm_name, create=False)
            off = self._HEADER_SIZE
            self._q_target = np.ndarray(
                (self._num_dofs,), dtype=np.float64,
                buffer=self._shm.buf[off : off + self._joint_size],
            )
            off += self._joint_size
            self._dof_pos = np.ndarray(
                (self._num_dofs,), dtype=np.float64,
                buffer=self._shm.buf[off : off + self._joint_size],
            )
            return True
        except FileNotFoundError:
            return False

    def read(self):
        import struct

        if not self._try_connect():
            return None
        try:
            ts = struct.unpack_from("d", self._shm.buf, 0)[0]
        except Exception:
            self._shm = None
            return None
        if ts <= self._last_timestamp:
            return None
        self._last_timestamp = ts
        return self._q_target.copy(), self._dof_pos.copy()

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None


def _run_viewer(shm_name: str, offset: float, diff_report_hz: float):
    import mujoco
    import mujoco.viewer
    import numpy as np

    g1 = _find_g1_mjcf()
    combined = _build_dual_g1_mjcf(g1, offset=offset)
    tmp = g1.parent / "_policy_output_viewer_tmp.xml"
    tmp.write_text(combined)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    data = mujoco.MjData(model)

    cmd_qpos, dof_names = _resolve_dof_qpos_addrs(model, "_cmd")
    act_qpos, _ = _resolve_dof_qpos_addrs(model, "_act")
    # Freejoint qpos stays at identity — the side-by-side offset is baked
    # into the frame_cmd/frame_act wrapper bodies in the MJCF, so the
    # pelvis sits at the frame origin (offset, 0, 0.793) by construction.
    mujoco.mj_kinematics(model, data)

    reader = _PolicyOutputShmReader(num_dofs=len(cmd_qpos), shm_name=shm_name)
    print(
        f"Policy-output viewer: waiting for SHM '{shm_name}' "
        f"(start `pi run wbt-teleop ...` first)",
        flush=True,
    )

    last_report = time.monotonic()
    report_interval = 1.0 / max(diff_report_hz, 0.01)
    last_q_target = None
    last_dof_pos = None
    # Diagnostics: count writer ticks seen between reports and track how
    # much dof_pos is moving between reads. If dof_pos delta is ~0 the
    # sim PD isn't advancing between policy calls (or the writer is
    # reading a static source), which makes the diff field meaningless.
    writes_seen = 0
    prev_dof_pos = None
    dof_pos_delta_accum = 0.0
    # Joints to print absolute values for — picks from the typical top-5.
    watch_names = [
        "right_knee_joint", "left_knee_joint", "left_elbow_joint",
        "waist_yaw_joint", "left_wrist_roll_joint",
    ]
    watch_idx = [dof_names.index(n) for n in watch_names if n in dof_names]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            reading = reader.read()
            if reading is not None:
                q_target, dof_pos = reading
                writes_seen += 1
                if prev_dof_pos is not None:
                    dof_pos_delta_accum += float(np.abs(dof_pos - prev_dof_pos).sum())
                prev_dof_pos = dof_pos.copy()
                last_q_target, last_dof_pos = q_target, dof_pos
                for i, addr in enumerate(cmd_qpos):
                    data.qpos[addr] = q_target[i]
                for i, addr in enumerate(act_qpos):
                    data.qpos[addr] = dof_pos[i]
                mujoco.mj_kinematics(model, data)
                viewer.sync()

            now = time.monotonic()
            if (
                last_q_target is not None
                and last_dof_pos is not None
                and now - last_report >= report_interval
            ):
                diff = np.abs(np.asarray(last_q_target) - np.asarray(last_dof_pos))
                top = np.argsort(diff)[-5:][::-1]
                parts = [f"{dof_names[j]}={diff[j]:+.3f}" for j in top]
                print(f"[diff] max={diff.max():.3f} rad top5: {' '.join(parts)}", flush=True)
                # Absolute values on watched joints so we can tell which side
                # the bias is on (command vs. achieved).
                abs_parts = [
                    f"{dof_names[j]} q_tgt={last_q_target[j]:+.3f} dof={last_dof_pos[j]:+.3f}"
                    for j in watch_idx
                ]
                print(f"[abs]  {' | '.join(abs_parts)}", flush=True)
                # Writer liveness + sim motion. writes_seen should bump by
                # ~report_interval*policy_hz (e.g. 50 at 50Hz/1s). dof_pos
                # sum-abs delta tells us whether sim state is actually
                # changing between ticks.
                print(
                    f"[live] writes_since_last_report={writes_seen} "
                    f"dof_pos_delta_sum={dof_pos_delta_accum:.4f} rad",
                    flush=True,
                )
                writes_seen = 0
                dof_pos_delta_accum = 0.0
                last_report = now

            time.sleep(0.002)

    reader.close()


def main(argv=None):
    _set_mujoco_gl_default()
    p = argparse.ArgumentParser(description="Real-time viewer of WBT policy output vs robot state.")
    p.add_argument(
        "--shm-name",
        default="holosoma_policy_output",
        help="Shared memory segment name (default: holosoma_policy_output).",
    )
    p.add_argument(
        "--offset",
        type=float,
        default=0.8,
        help="Side-by-side offset in meters (default: 0.8).",
    )
    p.add_argument(
        "--diff-report-hz",
        type=float,
        default=2.0,
        help="Console per-joint diff report rate (default: 2 Hz).",
    )
    args = p.parse_args(argv)
    _run_viewer(args.shm_name, args.offset, args.diff_report_hz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
