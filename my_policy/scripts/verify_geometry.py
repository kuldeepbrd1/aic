#!/usr/bin/env python3
"""
Verify all frame conventions and run the grasp deviation sweep (TAC-7).

Profiles lateral error at the port entrance across the full documented
grasp deviation envelope (±2 mm lateral, ±0.04 rad angular) and reports
how many of the 25 cases stay within the acceptance threshold.

Usage:
    python scripts/verify_geometry.py --connector sfp
    python scripts/verify_geometry.py --connector sc
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from my_policy.geometry import (
    apply_grasp_deviation,
    lateral_error_at_entrance,
    pose_to_matrix,
)

# ---------------------------------------------------------------------------
# Documented limits from the challenge specification
# ---------------------------------------------------------------------------
MAX_LATERAL_DEV_M: float = 0.002   # ±2 mm
MAX_ANGULAR_DEV_RAD: float = 0.04  # ±0.04 rad (~2.3°)

# Port catch radius × 0.8 — replace with connector-specific value once known.
# SFP port: ~1.5 mm catch radius → threshold ~1.2 mm
# SC  port: ~2.0 mm catch radius → threshold ~1.6 mm
ACCEPTANCE_THRESHOLD_BY_CONNECTOR: dict[str, float] = {
    "sfp": 0.0012,  # 1.2 mm  # TODO: validate against connector spec
    "sc": 0.0016,   # 1.6 mm  # TODO: validate against connector spec
}


def run_deviation_sweep(
    T_tcp_plug_nominal: np.ndarray,
    T_port_world: np.ndarray,
    acceptance_threshold_m: float,
    connector: str,
) -> bool:
    """Sweep ±5 lateral and ±5 angular samples and report lateral error.

    Args:
        T_tcp_plug_nominal: Nominal 4×4 transform of plug tip relative to TCP.
            Fill from challenge config before running.
        T_port_world: 4×4 port pose in world (base_link) frame.
            Fill from TF lookup or config before running.
        acceptance_threshold_m: Maximum allowed lateral error at entrance.
        connector: Connector type label for display.

    Returns:
        True if all 25 cases pass, False otherwise.
    """
    results = []
    for lat in np.linspace(-MAX_LATERAL_DEV_M, MAX_LATERAL_DEV_M, 5):
        for ang in np.linspace(-MAX_ANGULAR_DEV_RAD, MAX_ANGULAR_DEV_RAD, 5):
            T_perturbed = apply_grasp_deviation(T_tcp_plug_nominal, lat, ang)
            # Analytically propagate the deviation: assuming the approach
            # controller brings the plug tip to the port face, the TCP pose
            # in world becomes T_port_world @ inv(T_perturbed).
            # Replace with actual controller simulation for higher fidelity.
            T_tcp_world_at_entrance = T_port_world @ np.linalg.inv(T_perturbed)
            err = lateral_error_at_entrance(T_tcp_world_at_entrance, T_port_world)
            results.append(
                {
                    "lat_dev_mm": lat * 1000,
                    "ang_dev_deg": np.degrees(ang),
                    "lateral_error_mm": err * 1000,
                    "pass": err < acceptance_threshold_m,
                }
            )

    passed = sum(1 for r in results if r["pass"])
    n = len(results)
    print(
        f"\nConnector: {connector.upper()}  "
        f"threshold={acceptance_threshold_m*1000:.1f} mm  "
        f"result={passed}/{n} within threshold"
    )
    print(
        f"{'Lat dev (mm)':>14}  {'Ang dev (deg)':>14}  "
        f"{'Error (mm)':>12}  {'Pass':>6}"
    )
    for r in results:
        flag = "OK" if r["pass"] else "FAIL"
        print(
            f"{r['lat_dev_mm']:>14.1f}  {r['ang_dev_deg']:>14.2f}  "
            f"{r['lateral_error_mm']:>12.2f}  {flag:>6}"
        )

    if passed < n:
        print(
            f"\nWARNING: {n - passed} case(s) exceed acceptance threshold.\n"
            "Review approach controller alignment tolerance before TAC-10."
        )
    else:
        print("\nAll deviation cases within acceptance threshold.  TAC-7 PASS ✓")

    return passed == n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAC-7 geometry verification and grasp deviation sweep"
    )
    parser.add_argument(
        "--connector", choices=["sfp", "sc"], default="sfp",
        help="Connector type to verify (default: sfp)"
    )
    args = parser.parse_args()

    print(f"TAC-7 geometry verification — connector: {args.connector.upper()}")

    # TODO: Replace with actual transforms.
    #
    # T_tcp_plug: transform of the plug tip (cable end) relative to the gripper TCP.
    #   Obtain from TF tree at runtime:
    #     buffer.lookup_transform("gripper/tcp", f"{cable_name}/{plug_name}_link", Time())
    #   or from the cable URDF when the cable is attached to the gripper.
    #
    # T_port_world: transform of the port in the world (base_link) frame.
    #   Obtain from TF tree:
    #     buffer.lookup_transform("base_link",
    #         f"task_board/{target_module_name}/{port_name}_link", Time())
    #
    T_tcp_plug = np.eye(4)   # placeholder — replace with actual transform
    T_port_world = np.eye(4) # placeholder — replace with actual transform

    threshold = ACCEPTANCE_THRESHOLD_BY_CONNECTOR[args.connector]
    passed = run_deviation_sweep(T_tcp_plug, T_port_world, threshold, args.connector)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
