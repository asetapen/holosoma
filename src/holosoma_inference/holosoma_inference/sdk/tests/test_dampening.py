"""Unit tests for the shared command dampening shim."""

from __future__ import annotations

import os

import numpy as np
import pytest

from holosoma_inference.sdk.dampening import DampeningKnobs, Dampener


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
    yield


def _make_inputs(n=29):
    return (
        np.ones(n) * 2.0,           # cmd_q
        np.zeros(n),                # cmd_dq
        np.zeros(n),                # cmd_tau
        np.ones(n) * 100.0,         # kp
        np.ones(n) * 5.0,           # kd
    )


def test_identity_passthrough_with_defaults():
    d = Dampener()
    q, dq, tau, kp, kd = _make_inputs()
    qo, dqo, tauo, kpo, kdo = d.apply(q, dq, tau, kp, kd, None)
    assert np.allclose(qo, q)
    assert np.allclose(kpo, kp)
    assert np.allclose(kdo, kd)


def test_kp_kd_level_scaling(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_KP_LEVEL", "0.25")
    monkeypatch.setenv("HOLOSOMA_KD_LEVEL", "0.5")
    d = Dampener()
    q, dq, tau, kp, kd = _make_inputs()
    _, _, _, kpo, kdo = d.apply(q, dq, tau, kp, kd, None)
    assert np.allclose(kpo, kp * 0.25)
    assert np.allclose(kdo, kd * 0.5)


def test_slew_clamp_limits_delta_q(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_Q_SLEW_PER_TICK", "0.1")
    d = Dampener()
    q1 = np.zeros(4)
    q2 = np.array([1.0, -1.0, 0.05, 0.5])
    kp = np.zeros(4); kd = np.zeros(4); dq = np.zeros(4); tau = np.zeros(4)
    # First call establishes prev = q1 (but cap not applied on first call yet).
    d.apply(q1, dq, tau, kp, kd, None)
    qo, *_ = d.apply(q2, dq, tau, kp, kd, None)
    # Per-joint |Δ| must be ≤ 0.1.
    assert np.all(np.abs(qo - q1) <= 0.1 + 1e-9)
    # Signs preserved.
    assert qo[0] > 0 and qo[1] < 0


def test_q_limit_clip_respects_bounds(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_Q_LIMIT_SCALE", "1.0")
    d = Dampener(joint_limits_lo=[-1.0, -1.0], joint_limits_hi=[1.0, 1.0])
    q = np.array([5.0, -5.0])
    kp = np.zeros(2); kd = np.zeros(2); dq = np.zeros(2); tau = np.zeros(2)
    qo, *_ = d.apply(q, dq, tau, kp, kd, None)
    assert np.allclose(qo, [1.0, -1.0])


def test_q_limit_clip_scaled(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_Q_LIMIT_SCALE", "0.5")
    d = Dampener(joint_limits_lo=[-1.0, -1.0], joint_limits_hi=[1.0, 1.0])
    q = np.array([5.0, -5.0])
    kp = np.zeros(2); kd = np.zeros(2); dq = np.zeros(2); tau = np.zeros(2)
    qo, *_ = d.apply(q, dq, tau, kp, kd, None)
    # Scale=0.5 halves the range, centered on midpoint 0.
    assert np.allclose(qo, [0.5, -0.5])


def test_blend_alpha_zero_freezes_at_current(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_BLEND_ALPHA", "0.0")
    d = Dampener()
    q = np.array([10.0, 10.0])
    cur = np.array([1.5, -0.5])
    kp = np.zeros(2); kd = np.zeros(2); dq = np.zeros(2); tau = np.zeros(2)
    qo, *_ = d.apply(q, dq, tau, kp, kd, cur)
    assert np.allclose(qo, cur)


def test_blend_alpha_half_is_midpoint(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_BLEND_ALPHA", "0.5")
    d = Dampener()
    q = np.array([1.0, 1.0])
    cur = np.array([3.0, -1.0])
    kp = np.zeros(2); kd = np.zeros(2); dq = np.zeros(2); tau = np.zeros(2)
    qo, *_ = d.apply(q, dq, tau, kp, kd, cur)
    assert np.allclose(qo, [2.0, 0.0])


def test_reset_clears_slew_memory(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_Q_SLEW_PER_TICK", "0.1")
    d = Dampener()
    kp = np.zeros(2); kd = np.zeros(2); dq = np.zeros(2); tau = np.zeros(2)
    d.apply(np.zeros(2), dq, tau, kp, kd, None)
    d.apply(np.array([1.0, 1.0]), dq, tau, kp, kd, None)  # clamped
    d.reset()
    qo, *_ = d.apply(np.array([2.0, 2.0]), dq, tau, kp, kd, None)
    # After reset, no prev means no clamp on this call.
    assert np.allclose(qo, [2.0, 2.0])


def test_knobs_from_env_reads_all_five(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_KP_LEVEL", "0.7")
    monkeypatch.setenv("HOLOSOMA_KD_LEVEL", "0.8")
    monkeypatch.setenv("HOLOSOMA_Q_SLEW_PER_TICK", "0.05")
    monkeypatch.setenv("HOLOSOMA_Q_LIMIT_SCALE", "0.9")
    monkeypatch.setenv("HOLOSOMA_BLEND_ALPHA", "0.6")
    k = DampeningKnobs.from_env()
    assert k.kp_level == pytest.approx(0.7)
    assert k.kd_level == pytest.approx(0.8)
    assert k.q_slew_per_tick == pytest.approx(0.05)
    assert k.q_limit_scale == pytest.approx(0.9)
    assert k.blend_alpha == pytest.approx(0.6)


def test_knobs_from_env_rejects_bad_strings(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_KP_LEVEL", "notafloat")
    k = DampeningKnobs.from_env()
    assert k.kp_level == 1.0  # default wins
