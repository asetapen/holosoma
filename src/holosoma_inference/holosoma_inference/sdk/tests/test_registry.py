"""Unit tests for the SDK registry + HOLOSOMA_ROBOT_BACKEND override."""

from __future__ import annotations

import pytest


def test_builtin_fallback_includes_mujoco():
    from holosoma_inference.sdk import _load_builtin

    cls = _load_builtin("mujoco")
    assert cls is not None
    assert cls.__name__ == "MujocoInterface"


def test_builtin_fallback_includes_unitree_and_booster():
    from holosoma_inference.sdk import _load_builtin

    assert _load_builtin("unitree").__name__ == "UnitreeInterface"
    assert _load_builtin("booster").__name__ == "BoosterInterface"


def test_unknown_sdk_type_returns_none():
    from holosoma_inference.sdk import _load_builtin

    assert _load_builtin("does_not_exist") is None


def test_create_interface_env_override(monkeypatch):
    """HOLOSOMA_ROBOT_BACKEND=mujoco should override sdk_type='unitree'."""
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    pytest.importorskip("holosoma_retargeting")

    from holosoma_inference.config.config_values.robot import g1_29dof
    from holosoma_inference.sdk import create_interface

    monkeypatch.setenv("HOLOSOMA_ROBOT_BACKEND", "mujoco")
    monkeypatch.setenv("HOLOSOMA_MUJOCO_REAL_TIME", "0")
    cfg = g1_29dof
    if cfg.motor_kp is None or cfg.motor_kd is None:
        from dataclasses import replace as _replace

        cfg = _replace(
            cfg,
            motor_kp=tuple([100.0] * cfg.num_motors),
            motor_kd=tuple([5.0] * cfg.num_motors),
        )
    # sdk_type in config says "unitree" but env forces "mujoco".
    iface = create_interface(cfg, interface_str="lo", use_joystick=False)
    assert iface.__class__.__name__ == "MujocoInterface"
