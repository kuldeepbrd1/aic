# Sprint 1 Implementation Guide — `my_policy`

## AI for Industry Challenge · Qualification Playbook

> **Scope:** TAC-5, TAC-6, TAC-7, TAC-8, and TAC-9 — package scaffold,
> logging, geometry verification, force margin analysis, and the
> proximity-first floor policy. Stop before guarded insertion (TAC-10).

> **Correction vs original planning doc:** The original guide showed a
> standalone `PolicyNode(Node)` that directly subscribes to ROS topics.
> That is **not** how the AIC framework works. Policies are Python classes
> that inherit from `aic_model.policy.Policy` and implement a single
> `insert_cable()` method. All ROS plumbing (subscriptions, publishers,
> lifecycle, TF buffer) is handled by `aic_model`. See
> [docs/policy.md](../docs/policy.md) for the authoritative API.

---

## 0. Environment assumptions

| Item             | Value                                                        |
|------------------|--------------------------------------------------------------|
| ROS distro       | ROS 2 Kilted Kaiju                                           |
| Simulator        | Gazebo (official evaluation environment)                     |
| Policy rate      | 20 Hz (driven by `sleep_for(0.05)` inside `insert_cable()`) |
| Controller bridge | `aic_controller` (provided by challenge repo)               |
| Observations     | Via `get_observation()` → `Observation` msg containing 3× cameras, joint state, TCP pose/velocity, wrist F/T |
| Policy loader    | `aic_model` node, `policy` parameter = module path          |
| Python           | 3.10+                                                        |
| Key deps         | `rclpy`, `geometry_msgs`, `aic_model`, `aic_control_interfaces`, `numpy`, `scipy` |

---

## 1. TAC-5 — Package scaffold

### 1a. Package layout

```
my_policy/
  my_policy/
    __init__.py
    policy.py           ← ProximityFirstPolicy (TAC-9)
    state_machine.py    ← Phase enum (TAC-9)
    logger.py           ← EpisodeLogger, StepRecord (TAC-6)
    geometry.py         ← frame utilities (TAC-7)
    metrics.py          ← batch aggregator (TAC-6/TAC-11)
  scripts/
    verify_geometry.py  ← deviation sweep (TAC-7)
    force_margin_analysis.py ← abort threshold (TAC-8)
    replay.py           ← episode replay tool (TAC-6)
  resource/my_policy    ← ament marker (empty file)
  setup.py
  setup.cfg
  package.xml
  pixi.toml
```

### 1b. Policy class pattern (corrected)

Policies are **not** standalone ROS nodes. They are `Policy` subclasses
loaded dynamically into `aic_model`. The minimum required interface is:

```python
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

class MyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        # All sensor data via: obs = get_observation()
        #   obs.controller_state.tcp_pose   — current TCP pose (Pose)
        #   obs.wrist_wrench.wrench.force   — force (Vector3)
        #   obs.joint_state                 — JointState
        #   obs.left_image / center / right — sensor_msgs/Image
        #
        # Motion via: self.set_pose_target(move_robot, pose)
        # Timing via:  self.sleep_for(0.05)   # 20 Hz
        # TF via:      self._parent_node._tf_buffer.lookup_transform(...)
        return False  # True = success, False = failure/no-insertion
```

**`setup.py` has no `entry_points`** — the policy is loaded by module path,
not as a console script. The file checklist in the original guide's `setup.py`
example (`policy_node = my_policy.policy_node:main`) is incorrect for this
framework.

### 1c. Container validation checklist

Run these checks in order. Do not proceed past a failing step.

- [ ] `pixi reinstall my-policy` exits 0
- [ ] Start the eval container (see dev container README)
- [ ] Run the policy:
  ```bash
  pixi run ros2 run aic_model aic_model \
    --ros-args -p policy:=my_policy.policy.ProximityFirstPolicy
  ```
- [ ] Node `/aic_model` appears in `ros2 node list`
- [ ] Node reaches `active` lifecycle state within the engine timeout
- [ ] A full no-op episode completes without container errors
- [ ] `ros2 topic echo /aic_controller/pose_commands` shows timestamped
  `MotionUpdate` messages during the approach phases

> **Note:** The command `ros2 run my_policy policy_node` does **not** exist
> in this framework. The policy runs inside `aic_model`, not as its own node.
> The topic to monitor is `/aic_controller/pose_commands`, not `/aic_controller/target`.

**Done when:** all checks pass on a clean container launch.

---

## 2. TAC-6 — Per-episode logging and replay tooling

### 2a. Logger class — `my_policy/logger.py`

`EpisodeLogger` writes one CSV row per policy step and a JSON summary on
episode end. Key fields:

| Field | Source |
|---|---|
| `timestamp_ns` | `self.time_now().nanoseconds` |
| `tcp_actual_*` | `obs.controller_state.tcp_pose.position` |
| `wrench_f*` | `obs.wrist_wrench.wrench.force` |
| `wrench_magnitude` | `np.linalg.norm([fx, fy, fz])` |
| `insertion_depth_m` | reserved for TAC-10; always 0 in TAC-9 |

### 2b. Replay tool — `scripts/replay.py`

```bash
python scripts/replay.py <episode_id>
python scripts/replay.py <episode_id> --log-dir /tmp/aic_logs
```

Prints phase transitions, force profile, and the JSON summary for one run.
Samples every 20 steps; always prints rows with phase changes or stop reasons.

### 2c. Batch metrics — `python -m my_policy.metrics /tmp/aic_logs`

Reads all `*_summary.json` files and outputs:

```json
{
  "n_runs": 20,
  "mean_score": 12.4,
  "median_score": 13.0,
  "p10_score": 8.0,
  "classified_pct": 100.0,
  "failure_breakdown": { "startup_failure": 0, ... },
  "peak_force_max": 4.2
}
```

**Done when:** command prints a valid JSON report after a 5-run sanity batch.

---

## 3. TAC-7 — Geometry verification and grasp deviation envelope

### 3a. Frame utility module — `my_policy/geometry.py`

| Function | Purpose |
|---|---|
| `pose_to_matrix(pos, quat)` | `[x,y,z]` + `[x,y,z,w]` → 4×4 homogeneous |
| `matrix_to_pose(T)` | 4×4 → `(pos, quat)` |
| `apply_grasp_deviation(T, lat_m, ang_rad, axis)` | perturb plug-tip transform |
| `lateral_error_at_entrance(T_tcp, T_port)` | XY error in port frame |
| `standoff_pose_in_world(T_port, standoff_m)` | standoff target along port Z axis |

### 3b. Geometry verification — `scripts/verify_geometry.py`

```bash
python scripts/verify_geometry.py --connector sfp
python scripts/verify_geometry.py --connector sc
```

Sweeps a 5×5 grid of lateral (±2 mm) and angular (±0.04 rad) deviations and
reports how many of the 25 cases stay within the connector-specific catch
radius threshold.

**Before running:** replace the two `np.eye(4)` placeholders with real transforms:

```python
# T_tcp_plug — plug tip relative to gripper TCP.
# Obtain from TF at runtime:
#   buffer.lookup_transform("gripper/tcp",
#       f"{task.cable_name}/{task.plug_name}_link", Time())

# T_port_world — port in world (base_link) frame.
# Obtain from TF at runtime:
#   buffer.lookup_transform("base_link",
#       f"task_board/{task.target_module_name}/{task.port_name}_link", Time())
```

**Done when:** both connectors show ≥ 80 % PASS at maximum documented deviation.

---

## 4. TAC-8 — Force margin analysis

### `scripts/force_margin_analysis.py`

```bash
python scripts/force_margin_analysis.py --log-dir /tmp/aic_logs
```

**Workflow:**

1. Collect 20+ proximity-first episodes (no insertion, forces reflect approach noise).
2. Run the script — it recommends an abort threshold:
   ```
   threshold = clip(20 N − 2σ(peak forces), 5 N, 18 N)
   ```
3. Set `INSERTION_ABORT_THRESHOLD_N` in the TAC-10 state machine to the printed value.
4. Document: nominal force, noise σ, chosen threshold, safety margin.

**Done when:** estimated penalty rate < 5 % at the chosen threshold on a 20-run batch.

---

## 5. TAC-9 — Proximity-first policy (floor milestone)

### 5a. Phase enum — `my_policy/state_machine.py`

```
BOOTSTRAP → COARSE_APPROACH → FINE_APPROACH → ALIGNMENT_HOLD → SAFE_STOP
```

`GUARDED_INSERT`, `BACKOFF`, `RETRY` are defined but unused until TAC-10.

### 5b. Policy — `my_policy/policy.py`

**Class:** `ProximityFirstPolicy(Policy)`

**Tunable constants at the top of the file:**

| Constant | Default | Description |
|---|---|---|
| `COARSE_STANDOFF_M` | 0.050 | 50 mm approach standoff |
| `FINE_STANDOFF_M` | 0.005 | 5 mm entrance-plane standoff |
| `HOLD_DURATION_S` | 0.5 | Hold time at entrance |
| `STEP_DT_S` | 0.05 | Control loop period (20 Hz) |
| `POSITION_TOL_M` | 0.005 | Convergence tolerance (5 mm) |
| `TF_TIMEOUT_S` | 10.0 | Max wait for port TF frame |
| `MAX_APPROACH_STEPS` | 400 | Safety cap per phase (~20 s) |

**Port frame convention** (from `CheatCode` reference):
```
task_board/{task.target_module_name}/{task.port_name}_link
```

The standoff is placed along the port frame's **+Z axis** (insertion approach
direction). This correctly handles non-vertical ports without special-casing.

**`insert_cable()` return value:** always `False` — no insertion is attempted.
The engine scores proximity points independently of the return value.

### 5c. Floor milestone acceptance test

Run:
```bash
# 20-run SFP batch
pixi run ros2 run aic_model aic_model \
  --ros-args -p policy:=my_policy.policy.ProximityFirstPolicy

# Check metrics after batch:
python -m my_policy.metrics /tmp/aic_logs
```

**Acceptance criteria (ALL must pass before starting TAC-10):**

| Metric | Required |
|---|---|
| Proximity-positive rate | 100 % |
| Force penalty rate | 0 % |
| Off-limit contact rate | 0 % |
| Policy startup failure rate | 0 % |
| Classified failure rate | 100 % |

---

## 6. What comes next (out of scope)

- **TAC-10** — 8-phase controller with guarded insertion and backoff
- **TAC-11** — Core metrics dashboard (score aggregation, Tier 3 breakdown)
- **Sprint 2** — Residual visual alignment head
- **Sprint 3** — Force-conditioned local insertion expert

---

## 7. File checklist

```
my_policy/
  my_policy/
    __init__.py                ✓
    policy.py                  ✓  ProximityFirstPolicy(Policy) — TAC-9
    state_machine.py           ✓  Phase enum, PROXIMITY_PHASES — TAC-9
    logger.py                  ✓  EpisodeLogger, StepRecord — TAC-6
    geometry.py                ✓  frame utils, deviation helpers — TAC-7
    metrics.py                 ✓  batch metrics aggregator — TAC-6/TAC-11
  scripts/
    verify_geometry.py         ✓  deviation sweep + acceptance check — TAC-7
    force_margin_analysis.py   ✓  abort threshold recommendation — TAC-8
    replay.py                  ✓  episode replay tool — TAC-6
  resource/my_policy           ✓  ament marker
  setup.py                     ✓  no console_scripts entry needed
  setup.cfg                    ✓
  package.xml                  ✓
  pixi.toml                    ✓  mirrors aic_example_policies pattern
```

---

## 8. Corrections vs original planning document

| Location | Original | Corrected |
|---|---|---|
| Section 1a layout | `policy_node.py` as main entry | `policy.py` containing `ProximityFirstPolicy(Policy)` |
| Section 1b code | `class PolicyNode(Node)` with direct subs | `class ProximityFirstPolicy(Policy)` using callbacks |
| Section 1b topics | `/tcp_pose`, `/aic_controller/target` | `get_observation().controller_state.tcp_pose`, `move_robot(motion_update=...)` |
| Section 1b setup.py | `entry_points` registering `policy_node` | No `entry_points` — policy loaded by module path |
| Section 1c validation | `ros2 run my_policy policy_node` | `ros2 run aic_model aic_model --ros-args -p policy:=my_policy.policy.ProximityFirstPolicy` |
| Section 1c topic | `ros2 topic echo /aic_controller/target` | `ros2 topic echo /aic_controller/pose_commands` |
| Section 2b replay | Listed but not in final checklist | Created at `scripts/replay.py` |
| Section 5b policy code | Standalone Node with timer | `insert_cable()` blocking loop using `sleep_for()` |
