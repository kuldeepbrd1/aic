"""Frame transforms and geometry utilities for the AIC cable insertion task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose as RosPose


# ---------------------------------------------------------------------------
# Internal se(3) helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for v."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) → SE(3).

    xi = [v (3), omega (3)] where v is the translational component and omega
    is the rotation vector (axis × angle, ||omega|| = theta).
    """
    v, omega = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    T = np.eye(4)
    if theta < 1e-10:
        T[:3, 3] = v
        return T
    R = Rotation.from_rotvec(omega).as_matrix()
    T[:3, :3] = R
    omega_hat = omega / theta
    K = _skew(omega_hat)
    # Left Jacobian: J = I + (1-cosθ)/θ K̂ + (θ-sinθ)/θ K̂²
    J = (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta * K
        + (theta - np.sin(theta)) / theta * (K @ K)
    )
    T[:3, 3] = J @ v
    return T


def _se3_log(T: np.ndarray) -> np.ndarray:
    """Log map SE(3) → se(3).

    Returns xi = [v (3), omega (3)] with ||omega|| in [0, pi].
    """
    R, t = T[:3, :3], T[:3, 3]
    omega = Rotation.from_matrix(R).as_rotvec()   # ||omega|| guaranteed in [0, pi]
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.concatenate([t, omega])
    omega_hat = omega / theta
    K = _skew(omega_hat)
    half_theta = theta / 2.0
    # J^{-1} = α I + (1-α) ω̂ω̂ᵀ − (θ/2) K̂,  α = (θ/2) / tan(θ/2)
    alpha = half_theta / np.tan(half_theta)
    J_inv = (
        alpha * np.eye(3)
        + (1.0 - alpha) * np.outer(omega_hat, omega_hat)
        - half_theta * K
    )
    return np.concatenate([J_inv @ t, omega])


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Pose:
    """SE(3) pose with Lie group / algebra helpers.

    Internal storage: translation ``t`` (shape (3,)) and rotation ``r``
    (scipy ``Rotation``). Use class-methods to construct from common formats
    and ``as_*`` methods to convert.

    Twist convention (``as_twist`` / ``from_twist``): xi = [v (3), omega (3)]
    where v is the translational component and omega is the rotation vector
    (axis × angle). This matches the controller's ``tcp_error`` layout
    ``(x, y, z, rx, ry, rz)``.
    """

    t: np.ndarray   # translation, shape (3,)
    r: Rotation     # orientation

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Pose:
        """Return the SE(3) identity."""
        return cls(t=np.zeros(3), r=Rotation.identity())

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> Pose:
        """Construct from a 4×4 homogeneous transform matrix."""
        return cls(t=T[:3, 3].copy(), r=Rotation.from_matrix(T[:3, :3]))

    @classmethod
    def from_quat(cls, t: np.ndarray, q: np.ndarray) -> Pose:
        """Construct from translation and quaternion [x, y, z, w]."""
        return cls(t=np.asarray(t, dtype=float), r=Rotation.from_quat(q))

    @classmethod
    def from_rotvec(cls, t: np.ndarray, rotvec: np.ndarray) -> Pose:
        """Construct from translation and SO(3) rotation vector (axis × angle)."""
        return cls(t=np.asarray(t, dtype=float), r=Rotation.from_rotvec(rotvec))

    @classmethod
    def from_twist(cls, xi: np.ndarray) -> Pose:
        """Construct from an se(3) twist via the exponential map.

        xi = [v (3), omega (3)].
        """
        return cls.from_matrix(_se3_exp(np.asarray(xi, dtype=float)))

    @classmethod
    def from_ros(cls, msg: RosPose) -> Pose:
        """Construct from a ``geometry_msgs/Pose`` message."""
        t = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ])
        return cls.from_quat(t, q)

    # ------------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Return a 4×4 homogeneous transform matrix."""
        T = np.eye(4)
        T[:3, :3] = self.r.as_matrix()
        T[:3, 3] = self.t
        return T

    def as_quat(self) -> np.ndarray:
        """Return quaternion [x, y, z, w]."""
        return self.r.as_quat()

    def as_rotvec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (translation, rotvec) where rotvec = axis × angle."""
        return self.t.copy(), self.r.as_rotvec()

    def as_twist(self) -> np.ndarray:
        """Return the se(3) twist via the log map.

        Returns xi = [v (3), omega (3)] with ||omega|| in [0, pi].
        """
        return _se3_log(self.as_matrix())

    def as_ros(self) -> RosPose:
        """Return a ``geometry_msgs/Pose`` message."""
        from geometry_msgs.msg import Point, Pose as _RosPose, Quaternion
        q = self.r.as_quat()
        return _RosPose(
            position=Point(x=float(self.t[0]), y=float(self.t[1]), z=float(self.t[2])),
            orientation=Quaternion(
                x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
            ),
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def __matmul__(self, other: Pose) -> Pose:
        """Compose two poses: self ∘ other."""
        return Pose.from_matrix(self.as_matrix() @ other.as_matrix())

    def inv(self) -> Pose:
        """Return the inverse pose."""
        r_inv = self.r.inv()
        return Pose(t=-(r_inv.as_matrix() @ self.t), r=r_inv)

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Apply this pose to a point: R @ p + t."""
        return self.r.apply(p) + self.t

    # ------------------------------------------------------------------
    # Lie algebra helpers
    # ------------------------------------------------------------------

    def perturb(self, xi: np.ndarray) -> Pose:
        """Return self right-composed with exp(xi).

        Applies a small twist xi (in self's body frame) to produce a
        perturbed pose. xi = [v (3), omega (3)].
        """
        return self @ Pose.from_twist(xi)

    def log_diff(self, other: Pose) -> np.ndarray:
        """Twist from self to other expressed in self's frame.

        Returns log(self⁻¹ ∘ other) as xi = [v (3), omega (3)].
        """
        return (self.inv() @ other).as_twist()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        q = self.r.as_quat()
        return (
            f"Pose(t=[{self.t[0]:.4f}, {self.t[1]:.4f}, {self.t[2]:.4f}], "
            f"q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}])"
        )


# ---------------------------------------------------------------------------
# Standalone helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

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
