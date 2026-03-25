"""State machine phases and transition logic for the proximity-first policy."""

from enum import Enum


class Phase(str, Enum):
    BOOTSTRAP = "bootstrap"
    COARSE_APPROACH = "coarse_approach"
    FINE_APPROACH = "fine_approach"
    ALIGNMENT_HOLD = "alignment_hold"
    GUARDED_INSERT = "guarded_insert"  # reserved for TAC-10
    BACKOFF = "backoff"                # reserved for TAC-10
    RETRY = "retry"                    # reserved for TAC-10
    SAFE_STOP = "safe_stop"


# Phase sequence used by the TAC-9 proximity-first floor policy.
# TAC-10 will extend this with GUARDED_INSERT, BACKOFF, and RETRY.
PROXIMITY_PHASES = [
    Phase.BOOTSTRAP,
    Phase.COARSE_APPROACH,
    Phase.FINE_APPROACH,
    Phase.ALIGNMENT_HOLD,
    Phase.SAFE_STOP,
]
