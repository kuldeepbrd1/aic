"""
Proximity-first policy — Sprint 1 floor milestone (TAC-9).

Behaviour:
  BOOTSTRAP        — wait for first valid observation.
  COARSE_APPROACH  — move to 50 mm standoff in front of the port entrance.
  FINE_APPROACH    — slow down, move to 5 mm standoff.
  ALIGNMENT_HOLD   — hold pose for 0.5 s with low oscillation.
  SAFE_STOP        — publish stop, flush log, end episode.

This policy NEVER attempts insertion. It is the floor milestone that must
score proximity points (up to 25) on every trial before guarded insertion
(TAC-10) is implemented.

Acceptance criteria (must ALL pass before starting TAC-10):
  - Proximity-positive rate : 100 %
  - Force penalty rate      :   0 %
  - Off-limit contact rate  :   0 %
  - Startup failure rate    :   0 %
  - Classified failure rate : 100 %

Usage:
    pixi run ros2 run aic_model aic_model \\
        --ros-args -p policy:=my_policy.policy.ProximityFirstPolicy
"""

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .geometry import pose_to_matrix, standoff_pose_in_world
from .logger import EpisodeLogger, StepRecord
from .state_machine import Phase

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
COARSE_STANDOFF_M: float = 0.050   # 50 mm in front of port entrance
FINE_STANDOFF_M: float = 0.005     # 5 mm entrance-plane standoff
HOLD_DURATION_S: float = 0.5       # seconds to hold at entrance plane
STEP_DT_S: float = 0.05            # control loop period (20 Hz)
POSITION_TOL_M: float = 0.005      # convergence tolerance (5 mm)
TF_TIMEOUT_S: float = 10.0         # maximum wait for TF frames
MAX_APPROACH_STEPS: int = 400       # safety cap (~20 s at 20 Hz per phase)


class ProximityFirstPolicy(Policy):
    """Floor milestone: approach and hold at port entrance without inserting.

    Inherits from ``aic_model.policy.Policy``. Loaded dynamically by the
    ``aic_model`` node via the ``policy`` ROS parameter.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.logger = EpisodeLogger()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_tf(self, target_frame: str, source_frame: str) -> bool:
        """Wait up to TF_TIMEOUT_S for a transform to become available."""
        start = self.time_now()
        timeout = Duration(seconds=TF_TIMEOUT_S)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time()
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for TF '{source_frame}' → '{target_frame}'…"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"TF '{source_frame}' not available after {TF_TIMEOUT_S:.0f} s"
        )
        return False

    def _make_target_pose(
        self, port_tf, standoff_m: float
    ) -> tuple[Pose, np.ndarray]:
        """Build the ROS Pose for a standoff from the port, plus its position array."""
        port_pos = np.array([
            port_tf.translation.x,
            port_tf.translation.y,
            port_tf.translation.z,
        ])
        port_quat = np.array([
            port_tf.rotation.x,
            port_tf.rotation.y,
            port_tf.rotation.z,
            port_tf.rotation.w,
        ])
        T_port = pose_to_matrix(port_pos, port_quat)
        target_pos, target_quat = standoff_pose_in_world(T_port, standoff_m)
        pose = Pose(
            position=Point(
                x=float(target_pos[0]),
                y=float(target_pos[1]),
                z=float(target_pos[2]),
            ),
            orientation=Quaternion(
                x=float(target_quat[0]),
                y=float(target_quat[1]),
                z=float(target_quat[2]),
                w=float(target_quat[3]),
            ),
        )
        return pose, target_pos

    def _obs_tcp_pos(self, get_observation: GetObservationCallback) -> np.ndarray:
        tcp = get_observation().controller_state.tcp_pose.position
        return np.array([tcp.x, tcp.y, tcp.z])

    def _obs_force_mag(self, get_observation: GetObservationCallback) -> float:
        f = get_observation().wrist_wrench.wrench.force
        return float(np.linalg.norm([f.x, f.y, f.z]))

    def _log_step(
        self,
        episode_id: str,
        step: int,
        phase: Phase,
        get_observation: GetObservationCallback,
        target_pos: np.ndarray | None = None,
        stop_reason: str = "",
        insertion_depth_m: float = 0.0,
    ) -> None:
        obs = get_observation()
        tcp = obs.controller_state.tcp_pose.position
        f = obs.wrist_wrench.wrench.force
        force_mag = float(np.linalg.norm([f.x, f.y, f.z]))
        tp = target_pos if target_pos is not None else np.zeros(3)
        self.logger.log_step(
            StepRecord(
                timestamp_ns=self.time_now().nanoseconds,
                episode_id=episode_id,
                step=step,
                phase=phase.value,
                target_id="",
                tcp_target_x=float(tp[0]),
                tcp_target_y=float(tp[1]),
                tcp_target_z=float(tp[2]),
                tcp_actual_x=tcp.x,
                tcp_actual_y=tcp.y,
                tcp_actual_z=tcp.z,
                wrench_fx=f.x,
                wrench_fy=f.y,
                wrench_fz=f.z,
                wrench_magnitude=force_mag,
                stop_reason=stop_reason,
                insertion_depth_m=insertion_depth_m,
            )
        )

    def _run_approach(
        self,
        phase: Phase,
        episode_id: str,
        step: list[int],
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        target_pose: Pose,
        target_pos: np.ndarray,
    ) -> None:
        """Step toward target_pose until within POSITION_TOL_M or step cap hit."""
        send_feedback(phase.value)
        self.get_logger().info(
            f"{phase.value}: target=({target_pos[0]:.3f}, "
            f"{target_pos[1]:.3f}, {target_pos[2]:.3f})"
        )
        for _ in range(MAX_APPROACH_STEPS):
            self._log_step(episode_id, step[0], phase, get_observation, target_pos)
            step[0] += 1
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            tcp_pos = self._obs_tcp_pos(get_observation)
            if np.linalg.norm(target_pos - tcp_pos) < POSITION_TOL_M:
                self.get_logger().info(f"{phase.value}: converged")
                break
            self.sleep_for(STEP_DT_S)

    # ------------------------------------------------------------------
    # Policy entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Execute the proximity-first floor milestone.

        Returns False — no insertion is attempted. The caller (aic_engine)
        scores proximity points based on how close the plug tip gets to
        the port entrance, independent of a return value of True.
        """
        episode_id = task.id or f"ep_{self.time_now().nanoseconds}"
        self.logger.start_episode(episode_id)
        step = [0]  # mutable so helpers can increment it

        # --- Bootstrap: resolve port TF ---
        send_feedback(Phase.BOOTSTRAP.value)
        self.get_logger().info(
            f"ProximityFirstPolicy: task={task.id}  "
            f"port={task.port_name}  module={task.target_module_name}"
        )

        port_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link"
        )
        if not self._wait_for_tf("base_link", port_frame):
            self.logger.end_episode("startup_failure", "startup_failure")
            return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().error(f"Port TF lookup failed: {ex}")
            self.logger.end_episode("startup_failure", "startup_failure")
            return False

        port_tf = port_tf_stamped.transform

        # --- Coarse approach (50 mm standoff) ---
        coarse_pose, coarse_pos = self._make_target_pose(port_tf, COARSE_STANDOFF_M)
        self._run_approach(
            Phase.COARSE_APPROACH, episode_id, step,
            get_observation, move_robot, send_feedback,
            coarse_pose, coarse_pos,
        )

        # --- Fine approach (5 mm standoff) ---
        fine_pose, fine_pos = self._make_target_pose(port_tf, FINE_STANDOFF_M)
        self._run_approach(
            Phase.FINE_APPROACH, episode_id, step,
            get_observation, move_robot, send_feedback,
            fine_pose, fine_pos,
        )

        # --- Alignment hold ---
        send_feedback(Phase.ALIGNMENT_HOLD.value)
        self.get_logger().info(f"Alignment hold for {HOLD_DURATION_S:.1f} s")
        elapsed = 0.0
        while elapsed < HOLD_DURATION_S:
            self._log_step(
                episode_id, step[0], Phase.ALIGNMENT_HOLD,
                get_observation, fine_pos,
            )
            step[0] += 1
            self.set_pose_target(move_robot=move_robot, pose=fine_pose)
            self.sleep_for(STEP_DT_S)
            elapsed += STEP_DT_S

        # --- Safe stop ---
        send_feedback(Phase.SAFE_STOP.value)
        summary = self.logger.end_episode(
            stop_reason="proximity_hold_complete",
            failure_class="",
        )
        self.get_logger().info(f"Episode complete: {summary}")
        return False  # floor milestone — no insertion attempted
