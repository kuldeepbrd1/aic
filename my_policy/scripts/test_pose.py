"""Smoke tests for the Pose class (no ROS required).

Run with:
    python my_policy/scripts/test_pose.py
"""

import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from my_policy.geometry import Pose


ATOL = 1e-10


def check(name: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        raise AssertionError(name)


def test_identity() -> None:
    p = Pose.identity()
    check("identity matrix == eye(4)", np.allclose(p.as_matrix(), np.eye(4)))
    check("identity twist == zeros", np.allclose(p.as_twist(), np.zeros(6)))


def test_roundtrip_quat() -> None:
    t = np.array([0.1, -0.2, 0.5])
    q = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_quat()
    p = Pose.from_quat(t, q)
    check("quat roundtrip t", np.allclose(p.t, t))
    check("quat roundtrip q", np.allclose(p.as_quat(), q, atol=1e-14))


def test_roundtrip_matrix() -> None:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("zyx", [0.4, -0.2, 0.9]).as_matrix()
    T[:3, 3] = [1.0, -2.0, 0.5]
    p = Pose.from_matrix(T)
    check("matrix roundtrip", np.allclose(p.as_matrix(), T, atol=ATOL))


def test_twist_roundtrip_pure_translation() -> None:
    xi = np.array([0.1, -0.3, 0.7,  0.0, 0.0, 0.0])
    p = Pose.from_twist(xi)
    check("pure translation twist->matrix->twist", np.allclose(p.as_twist(), xi, atol=ATOL))


def test_twist_roundtrip_general() -> None:
    for i, xi in enumerate([
        np.array([0.1, -0.2,  0.3,  0.05, -0.1,  0.2 ]),   # small rotation
        np.array([0.0,  0.0,  0.0,  0.0,   0.0,  np.pi - 1e-6]),  # near-pi rotation
        np.array([1.0, -1.0,  0.5,  0.4,   0.3, -0.7 ]),   # moderate rotation
    ]):
        p = Pose.from_twist(xi)
        xi_back = p.as_twist()
        check(f"twist roundtrip case {i}", np.allclose(xi_back, xi, atol=1e-9))


def test_composition_and_inverse() -> None:
    T_a = np.eye(4)
    T_a[:3, :3] = Rotation.from_euler("z", 0.5).as_matrix()
    T_a[:3, 3] = [1.0, 0.0, 0.0]

    T_b = np.eye(4)
    T_b[:3, :3] = Rotation.from_euler("y", -0.3).as_matrix()
    T_b[:3, 3] = [0.0, 1.0, 0.0]

    a = Pose.from_matrix(T_a)
    b = Pose.from_matrix(T_b)
    ab = a @ b
    check("composition matrix", np.allclose(ab.as_matrix(), T_a @ T_b, atol=ATOL))

    a_inv = a.inv()
    eye = a @ a_inv
    check("pose @ inv == identity", np.allclose(eye.as_matrix(), np.eye(4), atol=ATOL))


def test_transform_point() -> None:
    p = Pose.from_quat([1.0, 2.0, 3.0], Rotation.from_euler("z", np.pi / 2).as_quat())
    pt = np.array([1.0, 0.0, 0.0])
    result = p.transform_point(pt)
    expected = np.array([1.0, 1.0 + 1.0, 3.0])   # 90° around z: (1,0)→(0,1), then +t
    check("transform_point", np.allclose(result, expected, atol=1e-12))


def test_perturb_and_log_diff() -> None:
    base = Pose.from_quat([0.5, -0.3, 0.2], Rotation.from_euler("xyz", [0.1, 0.2, -0.3]).as_quat())
    xi = np.array([0.01, -0.02, 0.005, 0.02, -0.01, 0.03])
    perturbed = base.perturb(xi)
    xi_back = base.log_diff(perturbed)
    check("perturb → log_diff roundtrip", np.allclose(xi_back, xi, atol=1e-12))


def test_repr() -> None:
    p = Pose.identity()
    s = repr(p)
    check("repr is a string", isinstance(s, str) and "Pose" in s)


def main() -> None:
    tests = [
        test_identity,
        test_roundtrip_quat,
        test_roundtrip_matrix,
        test_twist_roundtrip_pure_translation,
        test_twist_roundtrip_general,
        test_composition_and_inverse,
        test_transform_point,
        test_perturb_and_log_diff,
        test_repr,
    ]
    failed = 0
    for t in tests:
        print(f"{t.__name__}")
        try:
            t()
        except Exception:
            traceback.print_exc()
            failed += 1
    print()
    if failed:
        print(f"FAILED {failed}/{len(tests)}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
