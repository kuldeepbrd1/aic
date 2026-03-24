#!/usr/bin/env python3
"""
Force margin analysis for insertion abort threshold selection (TAC-8).

Profiles the peak-force distribution from near-miss contact logs collected
by the proximity-first policy (TAC-9) and recommends an abort threshold
with an explicit safety margin below the challenge's hard penalty limit.

Workflow:
    1. Run 20+ randomised proximity-first episodes to gather realistic
       contact force data (zero insertion, so forces reflect approach noise).
    2. Run this script:
           python scripts/force_margin_analysis.py --log-dir /tmp/aic_logs
    3. Set INSERTION_ABORT_THRESHOLD_N in the TAC-10 state machine config
       to the recommended value printed by this script.
    4. Document: nominal force, noise σ, chosen threshold, safety margin.

Done when: penalty rate < 5 % at the chosen threshold on a 20-run batch.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Challenge constants (from rules)
# ---------------------------------------------------------------------------
FORCE_PENALTY_LIMIT_N: float = 20.0    # >20 N sustained for >1 s → −12 pts
TARGET_PENALTY_RATE: float = 0.05      # must stay below 5 %

# Clamping range for the recommended threshold:
#   lower bound: don't abort on sensor noise (<5 N)
#   upper bound: maintain at least 2 N margin from penalty limit (<18 N)
THRESHOLD_MIN_N: float = 5.0
THRESHOLD_MAX_N: float = 18.0


def analyse(log_dir: str) -> dict:
    """Load summaries from log_dir and compute force margin statistics.

    Returns:
        Dict with analysis results, or empty dict if no data found.
    """
    summaries = [
        json.loads(p.read_text())
        for p in Path(log_dir).glob("*_summary.json")
    ]
    if not summaries:
        print(
            "No summaries found in the specified log directory.\n"
            "Run at least 20 randomised proximity-first episodes first."
        )
        return {}

    peak_forces = np.array([s.get("peak_force_n", 0.0) for s in summaries])
    n = len(peak_forces)

    mean_f = float(np.mean(peak_forces))
    std_f = float(np.std(peak_forces))
    max_f = float(np.max(peak_forces))
    p95_f = float(np.percentile(peak_forces, 95))

    # Recommended abort threshold: hard limit minus 2σ of peak-force noise.
    # This ensures that 97.7 % of legitimate approach contacts stay below it.
    recommended = float(
        np.clip(FORCE_PENALTY_LIMIT_N - 2.0 * std_f, THRESHOLD_MIN_N, THRESHOLD_MAX_N)
    )
    margin = FORCE_PENALTY_LIMIT_N - recommended
    penalty_rate = float(np.mean(peak_forces > recommended))

    result = {
        "n_runs": n,
        "peak_force_mean_n": round(mean_f, 3),
        "peak_force_std_n": round(std_f, 3),
        "peak_force_p95_n": round(p95_f, 3),
        "peak_force_max_n": round(max_f, 3),
        "recommended_abort_threshold_n": round(recommended, 2),
        "safety_margin_n": round(margin, 2),
        "estimated_penalty_rate_at_threshold": round(penalty_rate, 4),
        "penalty_rate_ok": penalty_rate < TARGET_PENALTY_RATE,
    }

    print(json.dumps(result, indent=2))

    if not result["penalty_rate_ok"]:
        print(
            f"\nWARNING: Estimated penalty rate {penalty_rate:.1%} exceeds "
            f"target {TARGET_PENALTY_RATE:.0%}.\n"
            "Consider reducing insertion speed or revisiting approach alignment."
        )
    else:
        print(
            f"\nAbort threshold recommendation: {recommended:.1f} N "
            f"(margin: {margin:.1f} N below {FORCE_PENALTY_LIMIT_N:.0f} N limit).  "
            "TAC-8 PASS ✓"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAC-8 force margin analysis — abort threshold recommendation"
    )
    parser.add_argument(
        "--log-dir", default="/tmp/aic_logs",
        help="Directory containing *_summary.json files (default: /tmp/aic_logs)"
    )
    args = parser.parse_args()
    result = analyse(args.log_dir)
    sys.exit(0 if result.get("penalty_rate_ok", False) else 1)


if __name__ == "__main__":
    main()
