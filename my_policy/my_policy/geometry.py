"""Frame transforms and geometry utilities for the AIC cable insertion task."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Convert position [x,y,z] + quaternion [x,y,z,w] to 4×4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and quaternion [x,y,z,w] from a 4×4 homogeneous matrix."""
    position = T[:3, 3].copy()
    quaternion = Rotation.from_matrix(T[:3, :3]).as_quat()
    return position, quaternion


def apply_grasp_deviation(
    T_tcp_plug: np.ndarray,
    lateral_dev_m: float,
    angular_dev_rad: float,
    axis: str = "x",
) -> np.ndarray:
    """Apply a grasp deviation to T_tcp_plug and return the perturbed transform.

    Used in TAC-7 sweeps to test robustness at documented limits (±2 mm / ±0.04 rad).

    Args:
        T_tcp_plug: 4×4 homogeneous transform of the plug tip relative to the TCP.
        lateral_dev_m: Lateral offset to apply (along `axis`).
        angular_dev_rad: Angular deviation about the Z axis.
        axis: Which axis to apply the lateral offset on ("x" or "y").

    Returns:
        Perturbed 4×4 transform.
    """
    dev = np.eye(4)
    if axis == "x":
        dev[0, 3] = lateral_dev_m
    elif axis == "y":
        dev[1, 3] = lateral_dev_m
    dev[:3, :3] = Rotation.from_euler("z", angular_dev_rad).as_matrix()
    return T_tcp_plug @ dev


def lateral_error_at_entrance(
    T_tcp_world: np.ndarray,
    T_port_world: np.ndarray,
) -> float:
    """Return the lateral distance between TCP and port centre in the port's XY plane.

    Projects both positions onto the plane perpendicular to the port's insertion
    axis (port Z) and returns the planar distance. A value below the port catch
    radius indicates the plug is aligned for insertion.

    Args:
        T_tcp_world: 4×4 TCP pose in the world (base_link) frame.
        T_port_world: 4×4 port pose in the world (base_link) frame.

    Returns:
        Lateral error in metres.
    """
    # Express TCP position in the port frame
    T_tcp_in_port = np.linalg.inv(T_port_world) @ T_tcp_world
    # XY components in port frame = lateral offset; Z = depth along insertion axis
    lateral = T_tcp_in_port[:2, 3]
    return float(np.linalg.norm(lateral))


def standoff_pose_in_world(
    T_port_world: np.ndarray,
    standoff_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute position and quaternion of a standoff point in front of the port.

    The standoff point is placed `standoff_m` along the port frame's +Z axis
    (the insertion approach direction) away from the port origin.

    Args:
        T_port_world: 4×4 port pose in the world (base_link) frame.
        standoff_m: Distance in metres from the port origin along its +Z axis.

    Returns:
        (position [x,y,z], quaternion [x,y,z,w]) in world frame.
    """
    approach_axis = T_port_world[:3, 2]  # port's Z axis in world frame
    target_pos = T_port_world[:3, 3] + standoff_m * approach_axis
    target_quat = Rotation.from_matrix(T_port_world[:3, :3]).as_quat()
    return target_pos, target_quat
