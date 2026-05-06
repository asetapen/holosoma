"""Unit tests for ``RetargetedTrackingMotionCommandSource``.

The retargeting adapter used to live inside
``WholeBodyTrackingPolicy._retarget_payload_to_motion_command``; the
service-level logic moved out of the policy class on 2026-05-06. These
tests exercise the adapter directly.

Covers:

* ``ClipMotionCommandSource`` is a no-op.
* ``RetargetedTrackingMotionCommandSource`` returns ``None`` when the raw
  source yields ``None`` (null transport).
* A one-shot payload produces a correctly shaped
  ``(motion_command, ref_quat_xyzw)`` tuple.
* An invalid root quaternion still returns a motion command but with
  ``ref_quat_xyzw = None`` (policy keeps its prior value).
* Retarget exceptions are swallowed and the adapter falls through.
* Missing ``urdf_path`` falls through once and never tries to construct
  a retargeter again.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from holosoma_inference.policies.motion_command_source import (
    ClipMotionCommandSource,
    RetargetedTrackingMotionCommandSource,
)
from holosoma_inference.policies.tracking_source import NullTrackingSource, TrackingPayload

_FAKE_MODULES = (
    "holosoma_retargeting",
    "holosoma_retargeting.src",
    "holosoma_retargeting.src.realtime_smpl_retargeter",
)


@pytest.fixture
def _cleanup_fake_retargeter():
    """Restore sys.modules after a test installs the retargeter stub.

    Without this, the fake modules leak into later tests that legitimately
    probe ``holosoma_retargeting`` (e.g. ``test_registry`` which looks at
    ``holosoma_retargeting.__file__``).
    """
    saved = {name: sys.modules.get(name) for name in _FAKE_MODULES}
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _install_fake_retargeter(returns):
    """Install a stub ``holosoma_retargeting.src.realtime_smpl_retargeter``.

    ``RetargetedTrackingMotionCommandSource`` constructs the retargeter
    lazily via ``from holosoma_retargeting.src.realtime_smpl_retargeter
    import SMPLRetargeter``. Tests that don't have the real package
    installed install a tree of minimal stub modules so that import line
    resolves without touching mink/mujoco. Pair with the
    ``_cleanup_fake_retargeter`` fixture so the stub does not leak.
    """
    mock = MagicMock()
    mock.retarget.return_value = returns
    pkg = types.ModuleType("holosoma_retargeting")
    src_pkg = types.ModuleType("holosoma_retargeting.src")
    realtime_mod = types.ModuleType("holosoma_retargeting.src.realtime_smpl_retargeter")
    realtime_mod.SMPLRetargeter = MagicMock(return_value=mock)
    pkg.src = src_pkg
    src_pkg.realtime_smpl_retargeter = realtime_mod
    sys.modules["holosoma_retargeting"] = pkg
    sys.modules["holosoma_retargeting.src"] = src_pkg
    sys.modules["holosoma_retargeting.src.realtime_smpl_retargeter"] = realtime_mod
    return mock


class _OneShotSource:
    def __init__(self, payload: TrackingPayload | None) -> None:
        self._payload = payload
        self._served = False

    def get_latest(self) -> TrackingPayload | None:
        if self._served or self._payload is None:
            return None
        self._served = True
        return self._payload


def _make_payload() -> TrackingPayload:
    transforms = np.zeros(24 * 7, dtype=np.float32)
    reshaped = transforms.reshape(24, 7)
    reshaped[:, 6] = 1.0  # identity quat qw
    return TrackingPayload(
        joint_names=[f"j{i}" for i in range(24)],
        joint_transforms=transforms,
        device_type="pico",
        tracking_quality=2,
        mode=1,
    )


class TestClipMotionCommandSource:
    def test_poll_returns_none(self) -> None:
        src = ClipMotionCommandSource()
        assert src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path="/tmp/any.xml") is None

    def test_reset_is_noop(self) -> None:
        ClipMotionCommandSource().reset()  # no exception


class TestRetargetedSource:
    def test_null_source_returns_none(self) -> None:
        src = RetargetedTrackingMotionCommandSource(tracking_source=NullTrackingSource())
        assert src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path="/tmp/fake.xml") is None

    def test_missing_urdf_falls_through(self) -> None:
        src = RetargetedTrackingMotionCommandSource(tracking_source=_OneShotSource(_make_payload()))
        assert src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path=None) is None
        # Further polls also return None without trying to reconstruct.
        assert src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path=None) is None

    @pytest.mark.usefixtures("_cleanup_fake_retargeter")
    def test_valid_payload_returns_motion_and_quat(self) -> None:
        _install_fake_retargeter((
            np.full(29, 0.1, dtype=np.float32),
            np.full(29, 0.2, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # wxyz identity
        ))
        src = RetargetedTrackingMotionCommandSource(tracking_source=_OneShotSource(_make_payload()))
        result = src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path="/tmp/fake.xml")
        assert result is not None
        motion_command, ref_quat = result
        assert motion_command.shape == (1, 58)
        assert ref_quat is not None
        assert ref_quat.shape == (1, 4)
        # wxyz identity -> xyzw (0,0,0,1).
        np.testing.assert_allclose(ref_quat[0], [0.0, 0.0, 0.0, 1.0])

    @pytest.mark.usefixtures("_cleanup_fake_retargeter")
    def test_invalid_root_quat_returns_motion_but_none_quat(self) -> None:
        _install_fake_retargeter((
            np.zeros(29, dtype=np.float32),
            np.zeros(29, dtype=np.float32),
            np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),  # non-finite quat
        ))
        src = RetargetedTrackingMotionCommandSource(tracking_source=_OneShotSource(_make_payload()))
        result = src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path="/tmp/fake.xml")
        assert result is not None
        motion_command, ref_quat = result
        assert motion_command.shape == (1, 58)
        assert ref_quat is None

    @pytest.mark.usefixtures("_cleanup_fake_retargeter")
    def test_retarget_exception_falls_through(self) -> None:
        mock = _install_fake_retargeter((np.zeros(29), np.zeros(29), np.zeros(4)))
        mock.retarget.side_effect = RuntimeError("IK divergence")
        src = RetargetedTrackingMotionCommandSource(tracking_source=_OneShotSource(_make_payload()))
        assert src.poll(num_dofs=29, rl_rate_hz=50.0, urdf_path="/tmp/fake.xml") is None

    def test_reset_clears_warn_state(self) -> None:
        src = RetargetedTrackingMotionCommandSource(tracking_source=NullTrackingSource())
        src._retargeter_runtime_warned = True
        src._retargeter_runtime_err_count = 42
        src._retargeter_runtime_last_warn_count = 10
        src.reset()
        assert not src._retargeter_runtime_warned
        assert src._retargeter_runtime_err_count == 0
        assert src._retargeter_runtime_last_warn_count == 0
