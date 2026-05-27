"""Unit tests for the dampening seam in BaseInterface (template-method).

These cover the absorbed dampener path independent of any real backend:
a fake _send_low_command_impl captures whatever the base class forwards,
so identity-passthrough vs. dampened paths can be diffed cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface
from holosoma_inference.sdk.dampening import DampeningConfig

N = 4


def _make_config(dampening: DampeningConfig | None = None) -> RobotConfig:
    return RobotConfig(
        robot_type="fake",
        robot="fake",
        default_dof_angles=tuple([0.0] * N),
        default_motor_angles=tuple([0.0] * N),
        motor2joint=tuple(range(N)),
        joint2motor=tuple(range(N)),
        dof_names=tuple(f"j{i}" for i in range(N)),
        dof_names_upper_body=(),
        dof_names_lower_body=tuple(f"j{i}" for i in range(N)),
        motor_kp=tuple([100.0] * N),
        motor_kd=tuple([5.0] * N),
        num_motors=N,
        num_joints=N,
        dampening=dampening,
    )


class _FakeInterface(BaseInterface):
    """Captures whatever _send_low_command_impl receives."""

    def __init__(self, robot_config: RobotConfig):
        super().__init__(robot_config, use_joystick=False)
        self.received: dict | None = None

    def get_low_state(self) -> np.ndarray:
        return np.zeros((1, 1))

    def _send_low_command_impl(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None, kp_override=None, kd_override=None):
        self.received = {
            "cmd_q": np.asarray(cmd_q),
            "cmd_dq": np.asarray(cmd_dq),
            "cmd_tau": np.asarray(cmd_tau),
            "dof_pos_latest": None if dof_pos_latest is None else np.asarray(dof_pos_latest),
            "kp_override": None if kp_override is None else np.asarray(kp_override),
            "kd_override": None if kd_override is None else np.asarray(kd_override),
        }

    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self):
        return 1.0

    @kp_level.setter
    def kp_level(self, value):
        pass

    @property
    def kd_level(self):
        return 1.0

    @kd_level.setter
    def kd_level(self, value):
        pass


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "HOLOSOMA_KP_LEVEL",
        "HOLOSOMA_KD_LEVEL",
        "HOLOSOMA_Q_SLEW_PER_TICK",
        "HOLOSOMA_Q_LIMIT_SCALE",
        "HOLOSOMA_BLEND_ALPHA",
    ):
        monkeypatch.delenv(name, raising=False)


def _send(iface: _FakeInterface, q=None, **kwargs):
    if q is None:
        q = np.array([0.5, -0.5, 1.0, -1.0])
    iface.send_low_command(
        cmd_q=q,
        cmd_dq=np.zeros(N),
        cmd_tau=np.zeros(N),
        kp_override=kwargs.get("kp_override"),
        kd_override=kwargs.get("kd_override"),
        dof_pos_latest=kwargs.get("dof_pos_latest"),
    )
    return iface.received


def test_no_dampening_passes_through_verbatim():
    iface = _FakeInterface(_make_config(dampening=None))
    assert iface._dampener is None
    q = np.array([0.5, -0.5, 1.0, -1.0])
    kp = np.array([100.0, 100.0, 100.0, 100.0])
    rec = _send(iface, q=q, kp_override=kp)
    assert np.array_equal(rec["cmd_q"], q)
    assert np.array_equal(rec["kp_override"], kp)


def test_no_dampening_does_not_apply_env_knobs(monkeypatch):
    """Env knobs are dampener-only. Without dampening, env vars are no-ops."""
    monkeypatch.setenv("HOLOSOMA_KP_LEVEL", "0.25")
    iface = _FakeInterface(_make_config(dampening=None))
    rec = _send(iface, kp_override=np.array([100.0] * N))
    assert np.allclose(rec["kp_override"], 100.0)


def test_dampening_default_config_is_identity_for_q():
    iface = _FakeInterface(_make_config(dampening=DampeningConfig()))
    assert iface._dampener is not None
    q = np.array([0.5, -0.5, 1.0, -1.0])
    rec = _send(iface, q=q)
    assert np.allclose(rec["cmd_q"], q)


def test_dampening_kp_kd_scaling():
    iface = _FakeInterface(_make_config(dampening=DampeningConfig(kp_level=0.5, kd_level=0.25)))
    rec = _send(iface, kp_override=np.array([100.0] * N), kd_override=np.array([8.0] * N))
    assert np.allclose(rec["kp_override"], 50.0)
    assert np.allclose(rec["kd_override"], 2.0)


def test_dampening_kp_kd_uses_robot_config_when_no_override():
    """When the caller omits kp_override, the dampener pulls motor_kp / motor_kd off RobotConfig and scales."""
    iface = _FakeInterface(_make_config(dampening=DampeningConfig(kp_level=0.1)))
    rec = _send(iface)
    assert np.allclose(rec["kp_override"], 10.0)


def test_dampening_slew_clamp_across_two_sends():
    iface = _FakeInterface(_make_config(dampening=DampeningConfig(q_slew_per_tick=0.1)))
    _send(iface, q=np.zeros(N))
    rec = _send(iface, q=np.array([1.0, -1.0, 0.05, 0.5]))
    assert np.all(np.abs(rec["cmd_q"]) <= 0.1 + 1e-9)
    assert rec["cmd_q"][0] > 0 and rec["cmd_q"][1] < 0


def test_dampening_blend_alpha_with_dof_pos_latest():
    iface = _FakeInterface(_make_config(dampening=DampeningConfig(blend_alpha=0.5)))
    q_target = np.array([2.0, 2.0, 2.0, 2.0])
    q_meas = np.array([0.0, 0.0, 0.0, 0.0])
    rec = _send(iface, q=q_target, dof_pos_latest=q_meas)
    assert np.allclose(rec["cmd_q"], 1.0)


def test_env_var_overrides_config(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_KP_LEVEL", "0.25")
    iface = _FakeInterface(_make_config(dampening=DampeningConfig(kp_level=1.0)))
    rec = _send(iface, kp_override=np.array([100.0] * N))
    assert np.allclose(rec["kp_override"], 25.0)


def test_env_var_only_applies_when_dampening_configured(monkeypatch):
    """HOLOSOMA_KP_LEVEL is dampener-internal; without DampeningConfig it does nothing."""
    monkeypatch.setenv("HOLOSOMA_KP_LEVEL", "0.25")
    iface = _FakeInterface(_make_config(dampening=None))
    rec = _send(iface, kp_override=np.array([100.0] * N))
    assert np.allclose(rec["kp_override"], 100.0)


def test_update_config_swaps_dampener():
    iface = _FakeInterface(_make_config(dampening=None))
    assert iface._dampener is None
    iface.update_config(_make_config(dampening=DampeningConfig(kp_level=0.5)))
    assert iface._dampener is not None
    rec = _send(iface, kp_override=np.array([100.0] * N))
    assert np.allclose(rec["kp_override"], 50.0)


def test_update_config_drops_dampener_when_disabled():
    iface = _FakeInterface(_make_config(dampening=DampeningConfig(kp_level=0.5)))
    assert iface._dampener is not None
    iface.update_config(_make_config(dampening=None))
    assert iface._dampener is None


def test_resolve_joint_limits_default_is_none():
    iface = _FakeInterface(_make_config(dampening=DampeningConfig()))
    assert iface._resolve_joint_limits(iface.robot_config) is None


def test_resolve_joint_limits_override_feeds_dampener():
    """A backend that overrides _resolve_joint_limits gets q_limit_scale clipping."""
    lo = np.array([-1.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0])

    class _LimitedInterface(_FakeInterface):
        def _resolve_joint_limits(self, robot_config):
            return lo, hi

    iface = _LimitedInterface(_make_config(dampening=DampeningConfig(q_limit_scale=1.0)))
    rec = _send(iface, q=np.array([5.0, -5.0, 0.5, -0.5]))
    assert np.allclose(rec["cmd_q"], [1.0, -1.0, 0.5, -0.5])
