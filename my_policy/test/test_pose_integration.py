"""Integration tests for the Pose class.

Tests are split into four groups:

  TestStandaloneFunctionCompat  — Pose output must be numerically identical to
                                  the existing standalone geometry functions so
                                  that callers can be migrated incrementally.

  TestSE3GroupAxioms            — Closure, associativity, identity, and inverse
                                  on a representative set of SE(3) poses.

  TestPolicyWorkflows           — Replicate the arithmetic performed in
                                  ProximityFirstPolicy._make_target_pose,
                                  lateral_error_at_entrance, and the TAC-7
                                  grasp-deviation sweep using Pose operations.

  TestROSRoundTrip              — from_ros / as_ros round-trip via
                                  geometry_msgs/Pose (auto-skipped without ROS).

No ROS runtime is required except for TestROSRoundTrip.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

try:
    import geometry_msgs.msg as _geometry_msgs_mod
    _HAS_GEOMETRY_MSGS = True
except ImportError:
    _geometry_msgs_mod = None
    _HAS_GEOMETRY_MSGS = False

from my_policy.geometry import (
    Pose,
    apply_grasp_deviation,
    lateral_error_at_entrance,
    matrix_to_pose,
    pose_to_matrix,
    standoff_pose_in_world,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ATOL = 1e-12


def _make_pose(euler_xyz, translation) -> Pose:
    """Helper: build a Pose from Euler angles and a translation vector."""
    return Pose.from_quat(
        np.array(translation, dtype=float),
        Rotation.from_euler("xyz", euler_xyz).as_quat(),
    )


# A varied set of (translation, euler_xyz) pairs that covers:
# pure translation, pure rotation, combined, small angles, large angles.
POSE_PARAMS = [
    ([0.0,  0.0,  0.0], [0.0,   0.0,   0.0  ]),   # identity
    ([1.0, -2.0,  0.5], [0.0,   0.0,   0.0  ]),   # pure translation
    ([0.0,  0.0,  0.0], [0.3,  -0.5,   1.1  ]),   # pure rotation
    ([0.1, -0.2,  0.5], [0.3,  -0.5,   1.1  ]),   # combined
    ([0.0,  0.0,  0.0], [0.01,  0.01, -0.01 ]),   # near-identity rotation
    ([0.0,  0.0,  1.0], [np.pi - 0.01, 0.0, 0.0]),  # near-π rotation
    ([0.5,  0.5,  0.5], [0.4,   0.3,  -0.7  ]),   # general
]

POSE_IDS = [
    "identity", "pure_translation", "pure_rotation",
    "combined", "near_identity_rot", "near_pi_rot", "general",
]


# ---------------------------------------------------------------------------
# Group 1: compatibility with standalone helper functions
# ---------------------------------------------------------------------------

class TestStandaloneFunctionCompat:
    """Pose must produce the same numbers as the existing module-level helpers."""

    @pytest.mark.parametrize("translation,euler", POSE_PARAMS, ids=POSE_IDS)
    def test_as_matrix_matches_pose_to_matrix(self, translation, euler):
        t = np.array(translation)
        q = Rotation.from_euler("xyz", euler).as_quat()
        expected = pose_to_matrix(t, q)
        result = Pose.from_quat(t, q).as_matrix()
        assert np.allclose(result, expected, atol=ATOL), (
            f"as_matrix differs from pose_to_matrix\ndiff={result - expected}"
        )

    @pytest.mark.parametrize("translation,euler", POSE_PARAMS, ids=POSE_IDS)
    def test_from_matrix_matches_matrix_to_pose(self, translation, euler):
        t = np.array(translation)
        q = Rotation.from_euler("xyz", euler).as_quat()
        T = pose_to_matrix(t, q)
        pos_expected, q_expected = matrix_to_pose(T)
        p = Pose.from_matrix(T)
        assert np.allclose(p.t, pos_expected, atol=ATOL)
        # Quaternions can differ by sign; compare rotation matrices instead.
        assert np.allclose(p.r.as_matrix(),
                           Rotation.from_quat(q_expected).as_matrix(), atol=ATOL)

    @pytest.mark.parametrize("translation,euler", POSE_PARAMS, ids=POSE_IDS)
    def test_standoff_matches_standalone(self, translation, euler):
        """Pose-based standoff computation matches standoff_pose_in_world."""
        T_port = pose_to_matrix(
            np.array(translation), Rotation.from_euler("xyz", euler).as_quat()
        )
        standoff_m = 0.05
        expected_pos, expected_q = standoff_pose_in_world(T_port, standoff_m)

        # Replicate via Pose: compose port pose with a local +Z translation.
        p_port = Pose.from_matrix(T_port)
        p_standoff = p_port @ Pose.from_quat([0.0, 0.0, standoff_m],
                                             Rotation.identity().as_quat())
        assert np.allclose(p_standoff.t, expected_pos, atol=ATOL)
        assert np.allclose(
            p_standoff.r.as_matrix(),
            Rotation.from_quat(expected_q).as_matrix(),
            atol=ATOL,
        )

    @pytest.mark.parametrize("translation,euler", POSE_PARAMS, ids=POSE_IDS)
    def test_lateral_error_matches_standalone(self, translation, euler):
        """Pose.inv() composition replicates lateral_error_at_entrance."""
        T_port = pose_to_matrix(
            np.array(translation), Rotation.from_euler("xyz", euler).as_quat()
        )
        # TCP is offset slightly from the port in world frame.
        T_tcp = pose_to_matrix(
            np.array(translation) + np.array([0.001, -0.002, 0.05]),
            Rotation.from_euler("xyz", euler).as_quat(),
        )
        expected_err = lateral_error_at_entrance(T_tcp, T_port)

        p_port = Pose.from_matrix(T_port)
        p_tcp = Pose.from_matrix(T_tcp)
        tcp_in_port = p_port.inv() @ p_tcp
        lateral = float(np.linalg.norm(tcp_in_port.t[:2]))
        assert np.isclose(lateral, expected_err, atol=ATOL)

    @pytest.mark.parametrize(
        "lat,ang,axis",
        [
            ( 0.001,  0.02, "x"),
            (-0.002, -0.04, "x"),
            ( 0.002,  0.00, "y"),
            ( 0.000,  0.04, "x"),
        ],
        ids=["lat+ang_x", "neg_lat+neg_ang_x", "lat_y", "ang_only"],
    )
    def test_perturb_matches_apply_grasp_deviation(self, lat, ang, axis):
        """Pose composition replicates apply_grasp_deviation."""
        T_nominal = pose_to_matrix(
            np.array([0.0, 0.0, 0.1]),
            Rotation.from_euler("z", 0.1).as_quat(),
        )
        expected = apply_grasp_deviation(T_nominal, lat, ang, axis)

        dev_t = np.array([lat, 0.0, 0.0] if axis == "x" else [0.0, lat, 0.0])
        dev_r = Rotation.from_euler("z", ang)
        p_result = Pose.from_matrix(T_nominal) @ Pose(t=dev_t, r=dev_r)
        assert np.allclose(p_result.as_matrix(), expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Group 2: SE(3) group axioms
# ---------------------------------------------------------------------------

class TestSE3GroupAxioms:
    """SE(3) must satisfy the four group axioms on Pose."""

    @pytest.fixture(params=list(zip(POSE_PARAMS, POSE_IDS)),
                    ids=POSE_IDS)
    def pose(self, request):
        (translation, euler), _ = request.param
        return _make_pose(euler, translation)

    def test_closure(self):
        """Composition of two Poses is a valid SE(3) element (det(R)==1)."""
        a = _make_pose([0.3, -0.5, 1.1], [0.1, -0.2, 0.5])
        b = _make_pose([0.4,  0.3, -0.7], [0.5,  0.5, 0.5])
        ab = a @ b
        R = ab.r.as_matrix()
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_associativity(self):
        """(a @ b) @ c == a @ (b @ c)."""
        a = _make_pose([0.3, -0.5,  1.1], [0.1, -0.2, 0.5])
        b = _make_pose([0.4,  0.3, -0.7], [0.5,  0.5, 0.5])
        c = _make_pose([0.1, -0.1,  0.2], [0.0,  0.3, 0.7])
        lhs = (a @ b) @ c
        rhs = a @ (b @ c)
        assert np.allclose(lhs.as_matrix(), rhs.as_matrix(), atol=ATOL)

    def test_identity_left(self, pose):
        """identity @ p == p."""
        result = Pose.identity() @ pose
        assert np.allclose(result.as_matrix(), pose.as_matrix(), atol=ATOL)

    def test_identity_right(self, pose):
        """p @ identity == p."""
        result = pose @ Pose.identity()
        assert np.allclose(result.as_matrix(), pose.as_matrix(), atol=ATOL)

    def test_inverse_left(self, pose):
        """p.inv() @ p == identity."""
        result = pose.inv() @ pose
        assert np.allclose(result.as_matrix(), np.eye(4), atol=1e-12)

    def test_inverse_right(self, pose):
        """p @ p.inv() == identity."""
        result = pose @ pose.inv()
        assert np.allclose(result.as_matrix(), np.eye(4), atol=1e-12)

    def test_double_inverse(self, pose):
        """inv(inv(p)) == p."""
        assert np.allclose(pose.inv().inv().as_matrix(), pose.as_matrix(), atol=ATOL)

    @pytest.mark.parametrize("translation,euler", POSE_PARAMS, ids=POSE_IDS)
    def test_transform_point_consistent_with_matrix(self, translation, euler):
        """transform_point(p) == (as_matrix() @ [p, 1])[:3]."""
        pose = _make_pose(euler, translation)
        pt = np.array([0.1, -0.3, 0.7])
        via_matrix = (pose.as_matrix() @ np.append(pt, 1.0))[:3]
        assert np.allclose(pose.transform_point(pt), via_matrix, atol=ATOL)


# ---------------------------------------------------------------------------
# Group 3: policy workflow replications
# ---------------------------------------------------------------------------

class TestPolicyWorkflows:
    """Replicate ProximityFirstPolicy._make_target_pose and TAC-7 sweep."""

    # Synthetic port and TCP transforms representative of the cable insertion
    # scene: port facing +Z at roughly arm-length, TCP nearby.
    _T_PORT = pose_to_matrix(
        np.array([0.6, 0.0, 0.4]),
        Rotation.from_euler("xyz", [0.0, -np.pi / 2, 0.0]).as_quat(),
    )
    _T_TCP_WORLD = pose_to_matrix(
        np.array([0.6, 0.001, 0.455]),
        Rotation.from_euler("xyz", [0.0, -np.pi / 2, 0.0]).as_quat(),
    )

    def test_make_target_pose_coarse(self):
        """Pose-based standoff matches standoff_pose_in_world at 50 mm."""
        standoff_m = 0.050
        expected_pos, expected_q = standoff_pose_in_world(self._T_PORT, standoff_m)

        p_port = Pose.from_matrix(self._T_PORT)
        p_target = p_port @ Pose.from_quat([0.0, 0.0, standoff_m],
                                           Rotation.identity().as_quat())
        assert np.allclose(p_target.t, expected_pos, atol=ATOL)
        assert np.allclose(
            p_target.r.as_matrix(),
            Rotation.from_quat(expected_q).as_matrix(),
            atol=ATOL,
        )

    def test_make_target_pose_fine(self):
        """Same check at 5 mm standoff (fine approach)."""
        standoff_m = 0.005
        expected_pos, expected_q = standoff_pose_in_world(self._T_PORT, standoff_m)

        p_port = Pose.from_matrix(self._T_PORT)
        p_target = p_port @ Pose.from_quat([0.0, 0.0, standoff_m],
                                           Rotation.identity().as_quat())
        assert np.allclose(p_target.t, expected_pos, atol=ATOL)

    def test_lateral_error_at_entrance_workflow(self):
        """Pose-based lateral error calculation matches the standalone function."""
        expected = lateral_error_at_entrance(self._T_TCP_WORLD, self._T_PORT)

        p_port = Pose.from_matrix(self._T_PORT)
        p_tcp = Pose.from_matrix(self._T_TCP_WORLD)
        tcp_in_port = p_port.inv() @ p_tcp
        lateral = float(np.linalg.norm(tcp_in_port.t[:2]))
        assert np.isclose(lateral, expected, atol=ATOL)

    def test_lateral_error_zero_for_aligned_tcp(self):
        """TCP exactly on the port axis → zero lateral error."""
        # Move TCP to standoff along port's Z axis (perfectly aligned).
        standoff_m = 0.05
        approach_axis = self._T_PORT[:3, 2]
        T_aligned = pose_to_matrix(
            self._T_PORT[:3, 3] + standoff_m * approach_axis,
            Rotation.from_matrix(self._T_PORT[:3, :3]).as_quat(),
        )
        err = lateral_error_at_entrance(T_aligned, self._T_PORT)
        assert err < 1e-12

    def test_tac7_deviation_sweep_via_pose(self):
        """TAC-7: Pose arithmetic produces numerically identical lateral errors to
        the standalone sweep across all 25 grasp-deviation cases (±2 mm / ±0.04 rad).

        Note: threshold acceptance (< 1.2 mm for SFP) is not checked here because
        it depends on the actual controller bringing the plug to the port face.
        That check lives in scripts/verify_geometry.py with real robot transforms.
        """
        T_tcp_plug = np.eye(4)   # nominal grasp: plug aligned with TCP
        T_port = self._T_PORT

        max_lat = 0.002
        max_ang = 0.04

        for lat in np.linspace(-max_lat, max_lat, 5):
            for ang in np.linspace(-max_ang, max_ang, 5):
                # --- standalone path (reference) ---
                T_perturbed = apply_grasp_deviation(T_tcp_plug, lat, ang)
                T_tcp_at_entrance = T_port @ np.linalg.inv(T_perturbed)
                expected_err = lateral_error_at_entrance(T_tcp_at_entrance, T_port)

                # --- Pose path ---
                dev_t = np.array([lat, 0.0, 0.0])
                dev_r = Rotation.from_euler("z", ang)
                p_tcp_plug = Pose.from_matrix(T_tcp_plug)
                p_perturbed = p_tcp_plug @ Pose(t=dev_t, r=dev_r)
                p_port = Pose.from_matrix(T_port)
                p_tcp_at_entrance = p_port @ p_perturbed.inv()
                pose_err = float(
                    np.linalg.norm((p_port.inv() @ p_tcp_at_entrance).t[:2])
                )

                assert np.isclose(pose_err, expected_err, atol=ATOL), (
                    f"lat={lat*1000:.1f}mm ang={np.degrees(ang):.2f}°: "
                    f"pose={pose_err*1000:.4f}mm standalone={expected_err*1000:.4f}mm"
                )

    def test_log_diff_gives_approach_direction(self):
        """log_diff from port pose to standoff pose has pure Z translation, no rotation."""
        standoff_m = 0.05
        p_port = Pose.from_matrix(self._T_PORT)
        p_target = p_port @ Pose.from_quat([0.0, 0.0, standoff_m],
                                           Rotation.identity().as_quat())
        xi = p_port.log_diff(p_target)  # body-frame twist: [v, omega]
        # In port's body frame the motion should be purely translational along Z.
        v, omega = xi[:3], xi[3:]
        assert np.isclose(v[0], 0.0, atol=1e-12), f"unexpected X translation: {v[0]}"
        assert np.isclose(v[1], 0.0, atol=1e-12), f"unexpected Y translation: {v[1]}"
        assert np.isclose(v[2], standoff_m, atol=1e-12), f"Z should be {standoff_m}: {v[2]}"
        assert np.allclose(omega, 0.0, atol=1e-12), f"unexpected rotation: {omega}"

    def test_perturb_log_diff_roundtrip_policy_scale(self):
        """perturb then log_diff recovers the original twist at typical policy scales."""
        p_base = Pose.from_matrix(self._T_PORT)
        # Twist at policy-relevant scale: ~5 mm translation, ~0.05 rad rotation.
        xi = np.array([0.005, -0.003, 0.050, 0.02, -0.01, 0.03])
        p_perturbed = p_base.perturb(xi)
        xi_recovered = p_base.log_diff(p_perturbed)
        assert np.allclose(xi_recovered, xi, atol=1e-12)


# ---------------------------------------------------------------------------
# Group 4: ROS message round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_GEOMETRY_MSGS, reason="geometry_msgs not available (no ROS)")
class TestROSRoundTrip:
    """from_ros / as_ros must be lossless inverses of each other.

    Skipped automatically when geometry_msgs is not importable (i.e. outside
    a sourced ROS 2 workspace).
    """

    geometry_msgs = _geometry_msgs_mod

    @pytest.mark.parametrize("translation,euler", POSE_PARAMS, ids=POSE_IDS)
    def test_from_ros_as_ros_roundtrip(self, translation, euler):
        gm = self.geometry_msgs
        q = Rotation.from_euler("xyz", euler).as_quat()
        t = np.array(translation)

        # Build a geometry_msgs/Pose message.
        msg_in = gm.Pose()
        msg_in.position.x, msg_in.position.y, msg_in.position.z = (
            float(t[0]), float(t[1]), float(t[2])
        )
        msg_in.orientation.x = float(q[0])
        msg_in.orientation.y = float(q[1])
        msg_in.orientation.z = float(q[2])
        msg_in.orientation.w = float(q[3])

        p = Pose.from_ros(msg_in)
        msg_out = p.as_ros()

        assert np.isclose(msg_out.position.x, msg_in.position.x, atol=ATOL)
        assert np.isclose(msg_out.position.y, msg_in.position.y, atol=ATOL)
        assert np.isclose(msg_out.position.z, msg_in.position.z, atol=ATOL)
        # Quaternion sign may flip; compare rotation matrices.
        R_in  = Rotation.from_quat([msg_in.orientation.x,  msg_in.orientation.y,
                                    msg_in.orientation.z,  msg_in.orientation.w]).as_matrix()
        R_out = Rotation.from_quat([msg_out.orientation.x, msg_out.orientation.y,
                                    msg_out.orientation.z, msg_out.orientation.w]).as_matrix()
        assert np.allclose(R_out, R_in, atol=ATOL)

    def test_from_ros_translation_only(self):
        gm = self.geometry_msgs
        msg = gm.Pose()
        msg.position.x, msg.position.y, msg.position.z = 1.0, -2.0, 3.0
        msg.orientation.w = 1.0   # identity quaternion

        p = Pose.from_ros(msg)
        assert np.allclose(p.t, [1.0, -2.0, 3.0], atol=ATOL)
        assert np.allclose(p.r.as_matrix(), np.eye(3), atol=ATOL)

    def test_as_ros_identity(self):
        gm = self.geometry_msgs
        msg = Pose.identity().as_ros()
        assert msg.position.x == 0.0
        assert msg.position.y == 0.0
        assert msg.position.z == 0.0
        # Identity quaternion: (x=0, y=0, z=0, w=1)
        assert np.isclose(abs(msg.orientation.w), 1.0, atol=ATOL)
        assert np.isclose(msg.orientation.x, 0.0, atol=ATOL)
        assert np.isclose(msg.orientation.y, 0.0, atol=ATOL)
        assert np.isclose(msg.orientation.z, 0.0, atol=ATOL)
