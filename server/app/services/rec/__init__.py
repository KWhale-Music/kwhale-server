"""Recommendation layer: scoring core and (later) candidate assembly."""

from .scoring import (
    DEFAULT_PROFILE,
    Candidate,
    Scored,
    apply_exploration,
    diversify,
    score_candidates,
)

__all__ = [
    "DEFAULT_PROFILE",
    "Candidate",
    "Scored",
    "apply_exploration",
    "diversify",
    "score_candidates",
]
