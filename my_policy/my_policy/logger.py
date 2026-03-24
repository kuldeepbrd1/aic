"""Episode logger — records all observations and policy decisions at policy rate."""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    timestamp_ns: int
    episode_id: str
    step: int
    phase: str
    target_id: str
    tcp_target_x: float = 0.0
    tcp_target_y: float = 0.0
    tcp_target_z: float = 0.0
    tcp_actual_x: float = 0.0
    tcp_actual_y: float = 0.0
    tcp_actual_z: float = 0.0
    tcp_vel_linear: float = 0.0
    wrench_fx: float = 0.0
    wrench_fy: float = 0.0
    wrench_fz: float = 0.0
    wrench_magnitude: float = 0.0
    cmd_type: str = ""
    stop_reason: str = ""
    # TAC-11: insertion depth from entrance plane
    insertion_depth_m: float = 0.0


class EpisodeLogger:
    """Logs every policy step to CSV; writes a JSON summary on episode end."""

    def __init__(self, log_dir: str = "/tmp/aic_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_id: str = ""
        self._steps: list[StepRecord] = []
        self._csv_writer = None
        self._csv_file = None

    def start_episode(self, episode_id: str) -> None:
        """Open a new CSV log for the given episode."""
        self._episode_id = episode_id
        self._steps = []
        csv_path = self.log_dir / f"{episode_id}.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=list(StepRecord.__dataclass_fields__.keys()),
        )
        self._csv_writer.writeheader()

    def log_step(self, record: StepRecord) -> None:
        """Append a step record to the in-memory list and CSV file."""
        self._steps.append(record)
        if self._csv_writer:
            self._csv_writer.writerow(asdict(record))

    def end_episode(self, stop_reason: str, failure_class: str = "") -> dict[str, Any]:
        """Flush CSV and write a JSON summary. Returns the summary dict.

        Args:
            stop_reason: Short string describing why the episode ended
                (e.g. "proximity_hold_complete", "force_limit_exceeded").
            failure_class: One of the 10 failure taxonomy buckets defined in
                metrics.py, or empty string on success.
        """
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        summary: dict[str, Any] = {
            "episode_id": self._episode_id,
            "total_steps": len(self._steps),
            "stop_reason": stop_reason,
            "failure_class": failure_class,
            "peak_force_n": max(
                (s.wrench_magnitude for s in self._steps), default=0.0
            ),
            "insertion_depth_m": (
                self._steps[-1].insertion_depth_m if self._steps else 0.0
            ),
            "phases_visited": list(
                dict.fromkeys(s.phase for s in self._steps)
            ),
        }

        summary_path = self.log_dir / f"{self._episode_id}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary
