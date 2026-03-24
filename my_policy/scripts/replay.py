#!/usr/bin/env python3
"""
Replay a single episode from its CSV log (TAC-6).

Prints a phase-annotated timeline with force and depth, then shows the
JSON summary. Useful for diagnosing why a specific run failed.

Usage:
    python scripts/replay.py <episode_id>
    python scripts/replay.py <episode_id> --log-dir /tmp/aic_logs
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def replay(episode_id: str, log_dir: str = "/tmp/aic_logs") -> bool:
    """Print a replay of one episode. Returns True on success."""
    log_dir = Path(log_dir)
    csv_path = log_dir / f"{episode_id}.csv"
    summary_path = log_dir / f"{episode_id}_summary.json"

    if not csv_path.exists():
        print(f"No log found for episode '{episode_id}' at {csv_path}")
        return False

    print(f"\n=== Replay: {episode_id} ===")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"Stop reason   : {summary['stop_reason']}")
        print(f"Failure class : {summary['failure_class'] or 'SUCCESS'}")
        print(f"Total steps   : {summary['total_steps']}")
        print(f"Peak force    : {summary['peak_force_n']:.2f} N")
        print(f"Insert depth  : {summary['insertion_depth_m'] * 1000:.1f} mm")
        print(f"Phases visited: {' → '.join(summary['phases_visited'])}")
    else:
        print("(no summary JSON found)")

    # Print only steps that have events (phase transitions or stop reasons),
    # plus every Nth step to give a progress trace.
    SAMPLE_EVERY = 20
    print(
        f"\n{'Step':>6}  {'Phase':<22}  {'Force N':>8}  "
        f"{'Depth mm':>9}  {'Event':<25}"
    )
    print("-" * 80)

    with open(csv_path) as f:
        reader = list(csv.DictReader(f))

    prev_phase = None
    for row in reader:
        step = int(row["step"])
        phase = row["phase"]
        force = float(row["wrench_magnitude"])
        depth = float(row["insertion_depth_m"]) * 1000
        stop = row["stop_reason"]

        # Decide whether to print this row
        phase_changed = phase != prev_phase
        has_event = bool(stop)
        sampled = step % SAMPLE_EVERY == 0

        if phase_changed or has_event or sampled:
            event = f"→ {phase}" if phase_changed else (stop or "")
            print(
                f"{step:>6}  {phase:<22}  {force:>8.2f}  "
                f"{depth:>9.1f}  {event:<25}"
            )
        prev_phase = phase

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAC-6 episode replay from CSV log"
    )
    parser.add_argument("episode_id", help="Episode ID (used as the log filename prefix)")
    parser.add_argument(
        "--log-dir", default="/tmp/aic_logs",
        help="Directory containing episode logs (default: /tmp/aic_logs)"
    )
    args = parser.parse_args()
    ok = replay(args.episode_id, args.log_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
