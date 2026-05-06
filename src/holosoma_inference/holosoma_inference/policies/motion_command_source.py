"""Motion-command source for WholeBodyTrackingPolicy.

Abstracts "where does this tick's motion_command_t + ref_quat_xyzw_t come
from" behind a single protocol, so the policy class stays
transport-agnostic and free of retargeting business logic.

Two implementations ship here:

* ``ClipMotionCommandSource`` — default no-op. The policy's ONNX clip
  drives ``motion_command_t`` (the policy itself already overwrites
  these fields from ``policy(input_feed)`` on each tick). Byte-identical
  to the pre-refactor policy when no external tracking is wired.

* ``RetargetedTrackingMotionCommandSource`` — wraps a raw
  ``TrackingSource`` (SMPLH joint transforms) and an ``SMPLRetargeter``,
  producing ``(motion_command_t, ref_quat_xyzw_t)`` per tick. Owns
  one-shot WARN / periodic re-WARN bookkeeping and lazy retargeter
  construction so imports (mink/mujoco) stay off the critical path
  until a payload actually arrives.

Callers inject a source into ``WholeBodyTrackingPolicy`` at construction
time. See the policy's ``rl_inference`` for the call site.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from loguru import logger

from holosoma_inference.policies.tracking_source import NullTrackingSource, TrackingSource


@runtime_checkable
class MotionCommandSource(Protocol):
    """Per-tick motion-command provider.

    ``poll`` is called once per inference step. It returns:
      * ``(motion_command, ref_quat_xyzw)`` when an external source has new
        data. ``motion_command`` is shape ``(1, 2*num_dofs)`` float32
        (joint_pos then joint_vel). ``ref_quat_xyzw`` is shape ``(1, 4)``
        float32 or ``None`` if only the motion command should be
        substituted (the policy keeps its prior ref_quat in that case).
      * ``None`` when no new data is available; the policy falls back to
        its ONNX-clip motion command.

    Implementations MUST be non-blocking.
    """

    def poll(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None) -> (
        tuple[np.ndarray, np.ndarray | None] | None
    ): ...

    def reset(self) -> None: ...


class ClipMotionCommandSource:
    """No-op source: the policy's ONNX-clip motion command is authoritative.

    This is the default when no external tracking is wired.
    """

    def poll(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None):
        return None

    def reset(self) -> None:
        return


class RetargetedTrackingMotionCommandSource:
    """Adapter: raw SMPLH ``TrackingSource`` -> (motion_command, ref_quat).

    Owns the retargeter + fallthrough bookkeeping so
    ``WholeBodyTrackingPolicy`` does not.
    """

    def __init__(
        self,
        tracking_source: TrackingSource | None = None,
        rewarn_every: int = 500,
        ik_iters: int = 4,
    ) -> None:
        self._tracking_source: TrackingSource = tracking_source or NullTrackingSource()
        self._retargeter = None  # type: ignore[var-annotated]
        self._retargeter_init_failed = False
        self._retargeter_runtime_warned = False
        self._retargeter_runtime_err_count = 0
        self._retargeter_runtime_last_warn_count = 0
        self._retargeter_runtime_rewarn_every = rewarn_every
        self._external_log_counter = 0
        self._ik_iters = ik_iters

    @property
    def tracking_source(self) -> TrackingSource:
        return self._tracking_source

    def reset(self) -> None:
        self._retargeter_runtime_warned = False
        self._retargeter_runtime_err_count = 0
        self._retargeter_runtime_last_warn_count = 0

    def _warn(self, reason: str) -> None:
        self._retargeter_runtime_err_count += 1
        if not self._retargeter_runtime_warned:
            logger.warning(
                "MotionCommandSource: {}: falling through to policy's clip motion_command. "
                "Subsequent failures suppressed; a periodic re-WARN will fire every {} frames.",
                reason,
                self._retargeter_runtime_rewarn_every,
            )
            self._retargeter_runtime_warned = True
            self._retargeter_runtime_last_warn_count = 1
            return
        delta = self._retargeter_runtime_err_count - self._retargeter_runtime_last_warn_count
        if delta >= self._retargeter_runtime_rewarn_every:
            logger.warning(
                "MotionCommandSource: retargeter still failing. {} total fallthroughs so far "
                "(last reason: {}).",
                self._retargeter_runtime_err_count,
                reason,
            )
            self._retargeter_runtime_last_warn_count = self._retargeter_runtime_err_count

    def _ensure_retargeter(self, rl_rate_hz: float, urdf_path: str | None):
        if self._retargeter is not None or self._retargeter_init_failed:
            return
        if not urdf_path:
            self._retargeter_init_failed = True
            logger.warning(
                "MotionCommandSource: received payload but urdf_path is unset; cannot construct "
                "SMPLRetargeter: falling through to clip motion_command."
            )
            return
        try:
            from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter  # noqa: PLC0415

            dt = 1.0 / float(rl_rate_hz)
            self._retargeter = SMPLRetargeter(
                urdf_path=urdf_path, dt=dt, max_ik_iters=self._ik_iters,
            )
        except Exception as exc:
            self._retargeter_init_failed = True
            logger.warning(
                "MotionCommandSource: failed to construct SMPLRetargeter "
                f"(urdf_path={urdf_path!r}): {exc!r}: falling through to clip motion_command."
            )

    def poll(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None):
        payload = self._tracking_source.get_latest()
        if payload is None:
            return None

        self._external_log_counter += 1
        if self._external_log_counter % 200 == 1:
            logger.debug(
                f"MotionCommandSource: payload #{self._external_log_counter} "
                f"device={payload.device_type!r} mode={payload.mode} "
                f"body_joints={len(payload.joint_names)} "
                f"hand_joints={len(payload.hand_joint_names)} "
                f"quality={payload.tracking_quality}"
            )

        self._ensure_retargeter(rl_rate_hz, urdf_path)
        if self._retargeter is None:
            return None

        try:
            transforms = np.asarray(payload.joint_transforms, dtype=np.float32).reshape(24, 7)
        except (ValueError, AttributeError) as exc:
            self._warn(f"joint_transforms reshape to (24, 7) failed ({exc!r})")
            return None

        try:
            q_joints, dq_joints, root_orn_wxyz = self._retargeter.retarget(transforms)
        except Exception as exc:
            self._warn(f"SMPLRetargeter.retarget raised {exc!r}")
            return None

        q_joints = np.asarray(q_joints, dtype=np.float32).reshape(-1)
        dq_joints = np.asarray(dq_joints, dtype=np.float32).reshape(-1)
        if q_joints.size != num_dofs or dq_joints.size != num_dofs:
            self._warn(
                f"retargeter returned unexpected dim (q={q_joints.size}, dq={dq_joints.size}, "
                f"expected={num_dofs})"
            )
            return None

        motion_command = np.concatenate([q_joints, dq_joints]).reshape(1, -1)

        root_orn_wxyz = np.asarray(root_orn_wxyz, dtype=np.float32).reshape(-1)
        if root_orn_wxyz.size != 4 or not np.all(np.isfinite(root_orn_wxyz)):
            self._warn(
                f"retargeter root_orn_wxyz invalid (size={root_orn_wxyz.size}); keeping "
                "previous ref_quat_xyzw_t"
            )
            return motion_command, None

        ref_quat_xyzw = np.array(
            [root_orn_wxyz[1], root_orn_wxyz[2], root_orn_wxyz[3], root_orn_wxyz[0]],
            dtype=np.float32,
        ).reshape(1, 4)
        return motion_command, ref_quat_xyzw
