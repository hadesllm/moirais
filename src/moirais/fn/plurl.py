# moirais.fn — function file (hadesllm/moirais)
"""Plurality voting (first-past-the-post)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plurality_vote(first_choices) -> DescriptiveResult:
    """You have power over your mind — not outside events. — Marcus Aurelius"""
    choices = np.asarray(first_choices, dtype=int).ravel()
    if len(choices) == 0:
        raise ValueError("first_choices must be non-empty.")
    candidates = np.unique(choices)
    counts = {int(c): int(np.sum(choices == c)) for c in candidates}
    winner = max(counts, key=counts.get)
    return DescriptiveResult(
        name="plurality_vote",
        value=winner,
        extra={"counts": counts, "n_voters": len(choices), "winner_pct": counts[winner] / len(choices)},
    )


plurl = plurality_vote


def cheatsheet() -> str:
    return "plurality_vote({}) -> Plurality (first-past-the-post) voting."
