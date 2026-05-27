"""Base interface for robot control.

The dampening seam lives here. ``send_low_command`` is concrete: it runs the
optional :class:`~holosoma_inference.sdk.dampening.Dampener` against the
incoming command, then forwards to ``_send_low_command_impl`` which each
backend implements with its own marshalling. New backends only override
``_send_low_command_impl`` and inherit the dampener wiring for free.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.dampening import Dampener


class BaseInterface(ABC):
    """
    Abstract base class for robot control interfaces.
    """

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.use_joystick = use_joystick
        # Initialize key state tracking for joystick
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}
        self._wc_key_map = self._default_wc_key_map()

        self._dampener: Dampener | None = self._build_dampener(robot_config)

    @abstractmethod
    def get_low_state(self) -> np.ndarray:
        """
        Get robot state as numpy array.

        Returns:
            np.ndarray with shape (1, 3+4+N+3+3+N) containing:
            [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        raise NotImplementedError

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Apply optional dampening, then forward to backend impl.

        Backends implement :meth:`_send_low_command_impl`; do not override
        this method.
        """
        if self._dampener is None:
            self._send_low_command_impl(
                cmd_q=cmd_q,
                cmd_dq=cmd_dq,
                cmd_tau=cmd_tau,
                dof_pos_latest=dof_pos_latest,
                kp_override=kp_override,
                kd_override=kd_override,
            )
            return

        kp_in = np.asarray(
            kp_override if kp_override is not None else self.robot_config.motor_kp,
            dtype=np.float64,
        )
        kd_in = np.asarray(
            kd_override if kd_override is not None else self.robot_config.motor_kd,
            dtype=np.float64,
        )
        knobs = self.robot_config.dampening.merged_with_env() if self.robot_config.dampening else None
        q_d, dq_d, tau_d, kp_d, kd_d = self._dampener.apply(
            cmd_q=np.asarray(cmd_q, dtype=np.float64),
            cmd_dq=np.asarray(cmd_dq, dtype=np.float64),
            cmd_tau=np.asarray(cmd_tau, dtype=np.float64),
            kp=kp_in,
            kd=kd_in,
            dof_pos_latest=dof_pos_latest,
            knobs=knobs,
        )
        self._send_low_command_impl(
            cmd_q=q_d,
            cmd_dq=dq_d,
            cmd_tau=tau_d,
            dof_pos_latest=dof_pos_latest,
            kp_override=kp_d,
            kd_override=kd_d,
        )

    @abstractmethod
    def _send_low_command_impl(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Backend-specific low-level command send.

        Receives commands AFTER the dampener has run (when configured).
        Backends must NOT re-apply self.kp_level / self.kd_level scaling
        when self._dampener is not None: the dampener owns kp/kd scaling
        in the dampened path. The kp_level/kd_level fields stay for
        back-compat in the non-dampened path.

        Args:
            cmd_q: target joint positions (N,)
            cmd_dq: target joint velocities (N,)
            cmd_tau: feedforward torques (N,)
            dof_pos_latest: latest joint positions (N,)
            kp_override: optional KP gains override (N,)
            kd_override: optional KD gains override (N,)
        """
        raise NotImplementedError

    def update_config(self, robot_config: RobotConfig):
        """
        Update the robot configuration and propagate to internal components.

        Override in subclasses that need to update internal SDK components
        when the config changes (e.g., after loading KP/KD from ONNX metadata).

        Args:
            robot_config: The new robot configuration.
        """
        self.robot_config = robot_config
        self._dampener = self._build_dampener(robot_config)

    def _build_dampener(self, robot_config: RobotConfig) -> Dampener | None:
        """Construct the per-interface Dampener when dampening is configured.

        Joint limits come from :meth:`_resolve_joint_limits`, which backends
        can override. Returns None when robot_config.dampening is None so
        send_low_command stays an identity transform for legacy consumers.
        """
        if robot_config.dampening is None:
            return None
        limits = self._resolve_joint_limits(robot_config)
        if limits is None:
            return Dampener()
        return Dampener(joint_limits_lo=limits[0], joint_limits_hi=limits[1])

    def _resolve_joint_limits(self, robot_config: RobotConfig) -> tuple[np.ndarray, np.ndarray] | None:
        """Best-effort joint-limit lookup for the q_limit_scale dampener knob.

        Default returns None (no MJCF available). UnitreeInterface overrides
        this to consult the holosoma_retargeting MJCF; backends without an
        MJCF concept (e.g. booster sdk2py) inherit the None default and
        leave HOLOSOMA_Q_LIMIT_SCALE as a no-op.
        """
        return None

    @abstractmethod
    def get_joystick_msg(self):
        raise NotImplementedError

    @abstractmethod
    def get_joystick_key(self, wc_msg=None):
        raise NotImplementedError

    def process_joystick_input(self, lin_vel_command, ang_vel_command, stand_command, upper_body_motion_active):
        """
        Process joystick input and update commands in a unified way.

        Args:
            lin_vel_command: np.ndarray, shape (1, 2)
            ang_vel_command: np.ndarray, shape (1, 1)
            stand_command: np.ndarray, shape (1, 1)
            upper_body_motion_active: bool

        Returns:
            (lin_vel_command, ang_vel_command, key_states): updated values
        """
        wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return lin_vel_command, ang_vel_command, self._key_states
        # Process stick input
        if getattr(wc_msg, "keys", 0) == 0 and not upper_body_motion_active:
            lx = getattr(wc_msg, "lx", 0.0)
            ly = getattr(wc_msg, "ly", 0.0)
            rx = getattr(wc_msg, "rx", 0.0)
            lin_vel_command[0, 1] = -(lx if abs(lx) > 0.1 else 0.0) * stand_command[0, 0]
            lin_vel_command[0, 0] = (ly if abs(ly) > 0.1 else 0.0) * stand_command[0, 0]
            ang_vel_command[0, 0] = -(rx if abs(rx) > 0.1 else 0.0) * stand_command[0, 0]
        # Process button input
        cur_key = self.get_joystick_key(wc_msg)
        self._last_key_states = self._key_states.copy()
        if cur_key:
            self._key_states[cur_key] = True
        else:
            self._key_states = dict.fromkeys(self._wc_key_map.values(), False)

        return lin_vel_command, ang_vel_command, self._key_states

    def _default_wc_key_map(self):
        """Default wireless controller key mapping."""
        return {
            1: "R1",
            2: "L1",
            3: "L1+R1",
            4: "start",
            8: "select",
            10: "L1+select",
            16: "R2",
            32: "L2",
            64: "F1",
            128: "F2",
            256: "A",
            264: "select+A",
            512: "B",
            520: "select+B",
            768: "A+B",
            1024: "X",
            1032: "select+X",
            1280: "A+X",
            1536: "B+X",
            2048: "Y",
            2304: "A+Y",
            2560: "B+Y",
            2056: "select+Y",
            3072: "X+Y",
            4096: "up",
            4097: "R1+up",
            4352: "A+up",
            4608: "B+up",
            4104: "select+up",
            5120: "X+up",
            6144: "Y+up",
            8192: "right",
            8193: "R1+right",
            8448: "A+right",
            9216: "X+right",
            10240: "Y+right",
            8200: "select+right",
            16384: "down",
            16392: "select+down",
            16385: "R1+down",
            16640: "A+down",
            16896: "B+down",
            17408: "X+down",
            18432: "Y+down",
            32768: "left",
            32769: "R1+left",
            32776: "select+left",
            33024: "A+left",
            33792: "X+left",
            34816: "Y+left",
        }

    @property
    @abstractmethod
    def kp_level(self):
        raise NotImplementedError

    @kp_level.setter
    @abstractmethod
    def kp_level(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def kd_level(self):
        raise NotImplementedError

    @kd_level.setter
    @abstractmethod
    def kd_level(self, value):
        raise NotImplementedError


def load_joint_limits_from_mjcf(robot_config: RobotConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort MJCF joint-limit lookup for the dampening clip knob.

    Resolves the same MJCF the retargeter uses and pulls per-joint hard
    limits in dof_names order. Returns (lo, hi) arrays indexed by joint
    or None when no MJCF is reachable (mujoco not installed,
    holosoma_retargeting absent on the host, MJCF parse error).
    Joints not found in the MJCF are left as +/-inf so the dampener's
    finite-only mask leaves them unclipped.
    """
    try:
        import mujoco
    except ImportError:
        return None

    candidates: list[Path] = []
    override = os.environ.get("HOLOSOMA_MUJOCO_MJCF")
    if override:
        candidates.append(Path(override))
    urdf = getattr(robot_config, "urdf_path", None)
    if urdf and urdf.endswith(".xml"):
        candidates.append(Path(urdf))
    try:
        import holosoma_retargeting

        candidates.append(
            Path(holosoma_retargeting.__file__).parent
            / "models"
            / robot_config.robot
            / f"{robot_config.robot_type}.xml"
        )
    except Exception as exc:
        logger.debug("holosoma_retargeting path probe failed: {}", exc)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:
            logger.debug("MJCF load failed at {}: {}", path, exc)
            continue
        lo = np.full(robot_config.num_joints, -np.inf)
        hi = np.full(robot_config.num_joints, np.inf)
        any_named = False
        for j_id, name in enumerate(robot_config.dof_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            any_named = True
            if bool(model.jnt_limited[jid]):
                lo[j_id] = float(model.jnt_range[jid][0])
                hi[j_id] = float(model.jnt_range[jid][1])
        if any_named:
            return lo, hi
    return None
