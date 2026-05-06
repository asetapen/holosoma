"""Policy output observer hook.

Core policy publishes (q_target, dof_pos) per tick via a lightweight
observer. Default implementation is a no-op, so the base policy has
zero knowledge of how (or whether) observers consume its output.

Transport-specific observers (shared memory, sockets, Prometheus gauges,
ROS topics, ...) live outside core and are injected at policy
construction time via the ``policy_output_observer`` constructor
parameter on ``WholeBodyTrackingPolicy``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PolicyOutputObserver(Protocol):
    """Receive per-tick policy output.

    ``on_tick`` is called once per policy step with the commanded set-point
    pre-Dampener (``q_target``) and the robot's reported joint positions
    (``dof_pos``). Both arrays are length ``num_dofs`` in the policy's
    joint order (identical to the ONNX ``dof_names``).

    Implementations MUST NOT block. The policy's control loop runs at
    50-60 Hz and cannot tolerate jitter; slow observers should offload
    work to a background queue.
    """

    def on_tick(self, q_target: np.ndarray, dof_pos: np.ndarray) -> None: ...

    def close(self) -> None: ...


class NullPolicyOutputObserver:
    """Default observer: swallow ticks.

    With this observer injected (the constructor default for
    ``WholeBodyTrackingPolicy``), per-tick policy output is discarded.
    Behavior is byte-identical to a policy with no observer wiring at all.
    """

    def on_tick(self, q_target: np.ndarray, dof_pos: np.ndarray) -> None:
        return

    def close(self) -> None:
        return
