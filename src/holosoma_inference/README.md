# Holosoma Inference

Policy inference for humanoid robot policies.

## Supported Policies

| Robot      | Locomotion | WBT |
|:----------:|:----------:|:---:|
| Unitree G1 | ✅         | ✅  |
| Booster T1 | ✅         | ❌  |

| Simulator               | Locomotion | WBT |
|:-----------------------:|:----------:|:---:|
| IsaacGym                | ✅         | ❌  |
| IsaacSim                | ✅         | ✅  |
| MuJoCo (inference only) | ✅         | ✅  |

- ✅ (full support)
- 🚧 (in progress/partial support)
- ❌ (no support)

## Quick Start

### Setup the Environment

```bash
cd ~/holosoma
bash scripts/setup_inference.sh             # Create a virtual environment with all dependencies
source scripts/source_inference_setup.sh    # Activate the virtual environment
```

### Choose Your Workflow

Select the appropriate workflow guide based on your setup:

#### Real Robot Deployment
- **[Real Robot Locomotion](docs/workflows/real-robot-locomotion.md)** - Run locomotion policies on physical Unitree G1 or Booster T1 robots
- **[Real Robot Whole Body Tracking](docs/workflows/real-robot-wbt.md)** - Run WBT policies on physical Unitree G1 robots

#### Simulation (MuJoCo)
- **[Sim-to-Sim Locomotion](docs/workflows/sim-to-sim-locomotion.md)** - Test locomotion policies in MuJoCo simulation
- **[Sim-to-Sim Whole Body Tracking](docs/workflows/sim-to-sim-wbt.md)** - Test WBT policies in MuJoCo simulation

Each workflow guide includes:
- Hardware/environment setup instructions
- Step-by-step commands
- Control references
- Deployment options (offboard/onboard/Docker)
- Troubleshooting tips

---

# Policy Controls

Commands for controlling policies during execution.

**Important**: All policy controls that use keyboard should be entered in the **policy terminal** (where you ran `run_policy.py`), not in the MuJoCo window. MuJoCo has separate controls for simulation (see workflow docs).

## General Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Stop the policy | `o` | B button |
| Set robot to default pose | `i` | Y button |
| Kill controller program | - | L1 (LB) + R1 (RB) |

## Locomotion (Velocity Tracking)

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Switch walking/standing | `=` | Start button |
| Adjust linear velocity | `w` `a` `s` `d` | Left stick |
| Adjust angular velocity | `q` `e` | Right stick |

**Default pose**: Standing pose

## Whole-Body Tracking

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Start motion clip | `m` | Select+A |

**Default pose**: Standing with raised arms

## Joystick-Only Features

- **Select button**: Switch between policies (when multiple policies are loaded)


# Configuration Overrides

## Loading ONNX Checkpoints from Wandb

You can load ONNX checkpoints directly from Wandb without manually downloading them first. This is useful for quickly testing models from training runs.

**Syntax:**
```bash
--task.model-path wandb://entity/project_name/run_id/model.onnx
```

**Example with G1 locomotion:**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path wandb://my-username/my-project/run-abc123/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface eth0
```

**Example with Wandb HTTPS URL:**
```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path https://wandb.ai/username/project/runs/abc123/files/model.onnx \
    --task.use-joystick \
    --task.interface eth0
```

The model will be automatically downloaded and cached locally. The entity is your Wandb username or organization name.

## Finding Your Network Interface

The `--task.interface` parameter specifies which network interface to use for communicating with the robot. The correct interface name varies by computer and network card.

**Common interface names:**
- `eth0` - Common Ethernet interface name
- `enp0s31f6` - Modern Linux Ethernet naming
- `lo` - Loopback interface (for sim2sim)

**To find your interface name:**
```bash
ifconfig
```

Look for the interface connected to your robot's network. For real robot deployments, use the interface with an IP address on the same subnet as your robot. For sim2sim deployments, use `lo` (loopback).

## Observation History Length (> 1)

If a policy was trained with stacked observations (e.g., history length 4), you must pass the same history length at inference time so the observation tensor matches the model's expected input size.

Example:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path <path-to-model>.onnx \
    --task.interface eth0 \
    --observation.history_length_dict.actor_obs=4
```

The override updates the `actor_obs` buffer before the ONNX session is initialized, so any policy (locomotion or WBT) can run with longer observation histories as long as the underlying model was trained that way.


## Input Sources

The policy supports two independent input channels that can each be set to `keyboard`, `joystick`, or `ros2`:

| Channel | What it controls | CLI flag |
|---------|-----------------|----------|
| `velocity_input` | Continuous velocity commands (linear/angular) | `--task.velocity-input` |
| `state_input` | Discrete commands: start/stop, walk/stand, kp tuning, policy switching | `--task.state-input` |

### Shortcuts (backwards compatible)

For the common case where both channels use the same source:

```bash
--task.use-joystick    # Sets both channels to joystick
--task.use-keyboard    # Sets both channels to keyboard (this is the default)
```

### Mixed input examples

**ROS2 `cmd_vel` with keyboard commands** — velocity from ROS2, operator uses keyboard for start/stop:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path <path-to-model>.onnx \
    --task.velocity-input ros2 \
    --task.state-input keyboard \
    --task.interface eth0
```

**Fully autonomous** — both channels from ROS2:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path <path-to-model>.onnx \
    --task.velocity-input ros2 \
    --task.state-input ros2 \
    --task.interface eth0
```

### ROS2 topics

| Topic | Message type | Channel | Description |
|-------|-------------|---------|-------------|
| `cmd_vel` | `geometry_msgs/TwistStamped` | `velocity_input` | Linear (x, y) and angular (z) velocity, clamped to [-1, 1] |
| `holosoma/state_input` | `std_msgs/String` | `state_input` | Text commands: `walk`, `stand`, `start`, `stop`, `init` |

Topic names are configurable via `--task.ros-cmd-vel-topic` and `--task.ros-state-input-topic`.

**Note**: Shortcut flags (`--task.use-joystick`, `--task.use-keyboard`) cannot be combined with `--task.velocity-input` or `--task.state-input`.

## Overriding Control Gains

By default, control gains (kp/kd) are loaded from ONNX model metadata. You can override these values in your configuration:

### G1 29-DOF with Custom Control Gains

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --robot.motor-kp 40.2 99.1 40.2 99.1 28.5 28.5 40.2 99.1 40.2 99.1 28.5 28.5 40.2 28.5 28.5 14.3 14.3 14.3 14.3 14.3 16.8 16.8 14.3 14.3 14.3 14.3 14.3 16.8 16.8 \
    --robot.motor-kd 2.6 6.3 2.6 6.3 1.8 1.8 2.6 6.3 2.6 6.3 1.8 1.8 2.6 1.8 1.8 0.9 0.9 0.9 0.9 0.9 1.1 1.1 0.9 0.9 0.9 0.9 0.9 1.1 1.1
```

### T1 29-DOF with Custom Control Gains

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --robot.motor-kp 5.0 5.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 200.0 200.0 200.0 200.0 200.0 50.0 50.0 200.0 200.0 200.0 200.0 50.0 50.0 \
    --robot.motor-kd 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 5.0 5.0 5.0 5.0 5.0 3.0 3.0 5.0 5.0 5.0 5.0 3.0 3.0
```

**Note**: When control gains are not specified, they will be automatically loaded from the ONNX model metadata. This is the recommended approach as it ensures the gains match those used during training.

## Unitree MP Driver (subprocess-isolated backend)

The Unitree C++ binding (`unitree_interface`) ships its own bundled CycloneDDS and conflicts with `rclpy`'s DDS when both share an address space. The `unitree_mp` SDK runs the real `UnitreeInterface` in a spawned child process and exposes the same `BaseInterface` API to the parent over RPC-over-queues, so ROS2 inputs and the Unitree binding can coexist in the same deployment.

Use it by setting `sdk_type="unitree_mp"` on the robot config (instead of `"unitree"`). The proxy is a drop-in replacement — the rest of the inference pipeline (`run_policy.py`, ROS2 inputs, joystick, kp/kd overrides, `update_config`) is unchanged.

**When to use**:
- Running with `--task.velocity-input ros2` or `--task.state-input ros2` on a real Unitree G1.
- Any deployment that imports both `rclpy` and `unitree_interface` in the same process.

**When to skip**: pure keyboard / joystick deployments without ROS2. The single-process `"unitree"` backend is lighter (no IPC hop per command) and remains the default.

## Command Dampening

`BaseInterface.send_low_command` runs every outgoing command through an optional `Dampener` shim before the backend marshals it to the wire. The shim is shared across `unitree`, `unitree_mp`, and `booster` backends and is intended for safely de-rating policy output during bring-up, hardware debugging, or cautious first-runs.

Dampening is opt-in: with `RobotConfig.dampening = None` (the default), `send_low_command` is an identity transform and existing behavior is preserved.

### Enabling dampening

Attach a `DampeningConfig` to your robot config:

```python
from holosoma_inference.sdk.dampening import DampeningConfig

robot_config.dampening = DampeningConfig(
    kp_level=0.5,           # halve all KP gains
    kd_level=0.5,           # halve all KD gains
    q_slew_per_tick=0.02,   # cap |Δq_target| per call (rad)
    q_limit_scale=0.8,      # clip q_target to 80% of each joint's hard range
    blend_alpha=1.0,        # 1.0 = pass-through, 0.0 = freeze at measured q
)
```

Joint hard limits are auto-resolved from the same MJCF the retargeter uses (override with the `HOLOSOMA_MUJOCO_MJCF` env var). Backends without an MJCF (e.g. Booster) leave `q_limit_scale` as a no-op.

### Runtime overrides (env vars)

Each knob is re-read from the environment on every `send_low_command`, so operators can toggle dampening live without restarting the driver. Env vars override the `DampeningConfig` value for that knob; unset vars fall through to the config.

| Env var | Default | Effect |
|---------|---------|--------|
| `HOLOSOMA_KP_LEVEL` | `1.0` | Multiplicative scale on KP |
| `HOLOSOMA_KD_LEVEL` | `1.0` | Multiplicative scale on KD |
| `HOLOSOMA_Q_SLEW_PER_TICK` | unset (off) | Max \|Δq_target\| per call, applied per-joint |
| `HOLOSOMA_Q_LIMIT_SCALE` | unset (off) | Scale `[0, 1]` against per-joint hard limits when clipping |
| `HOLOSOMA_BLEND_ALPHA` | `1.0` | `q_send = α·q_target + (1−α)·q_current`; requires `dof_pos_latest` |

Order of operations inside the shim: blend → joint-limit clip → slew clamp.

### Send-command logging

For post-run analysis, set `HOLOSOMA_SEND_LOG=1` to write every Nth outgoing command (and optionally the corresponding measured state) as JSONL. See `holosoma_inference/sdk/send_log.py` for the full env-var surface.
