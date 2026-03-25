"""Aggregate run summaries into a batch metrics report.

Usage:
    python -m my_policy.metrics /tmp/aic_logs
"""

import json
import sys
from pathlib import Path


# The 10 failure taxonomy buckets used across all sprint analyses.
FAILURE_CLASSES = [
    "startup_failure",
    "wrong_target",
    "entrance_miss",
    "bad_orientation",
    "early_contact_abort",
    "force_penalty",
    "off_limit_collision",
    "false_insertion_confidence",
    "recovery_loop",
    "sc_generalization_failure",
]


def compute_batch_metrics(log_dir: str = "/tmp/aic_logs") -> dict:
    """Load all *_summary.json files from log_dir and compute aggregate stats.

    Returns:
        Dict with keys:
            n_runs, mean_score, median_score, p10_score,
            classified_pct, failure_breakdown, peak_force_max.
        Empty dict if no summaries are found.
    """
    summaries = [
        json.loads(p.read_text())
        for p in Path(log_dir).glob("*_summary.json")
    ]
    if not summaries:
        return {}

    scores = sorted(s.get("score_total", 0.0) for s in summaries)
    n = len(scores)
    p10_idx = max(0, int(n * 0.1) - 1)

    classified = sum(
        1
        for s in summaries
        if s.get("failure_class") or s.get("stop_reason") == "success"
    )

    failure_breakdown = {fc: 0 for fc in FAILURE_CLASSES}
    for s in summaries:
        fc = s.get("failure_class", "")
        if fc in failure_breakdown:
            failure_breakdown[fc] += 1

    return {
        "n_runs": n,
        "mean_score": sum(scores) / n,
        "median_score": scores[n // 2],
        "p10_score": scores[p10_idx],
        "classified_pct": classified / n * 100,
        "failure_breakdown": failure_breakdown,
        "peak_force_max": max(s.get("peak_force_n", 0.0) for s in summaries),
    }


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/aic_logs"
    metrics = compute_batch_metrics(log_dir)
    print(json.dumps(metrics, indent=2))
