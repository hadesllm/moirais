# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Approval voting."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def approval_vote(approval_matrix) -> DescriptiveResult:
    """Approval voting: each voter approves one or more candidates.

    :param approval_matrix: (n_voters x n_candidates) binary matrix (1=approve).
    :return: DescriptiveResult with winner index and approval counts.

    References
    ----------
    Armstrong (2014), Ch 1. Brams & Fishburn (1978).

    .. epigraph:: "It's a trap!" -- Admiral Ackbar, Star Wars
    """
    A = np.asarray(approval_matrix, dtype=int)
    if A.ndim != 2:
        raise ValueError("approval_matrix must be 2D (voters x candidates).")
    n_voters, n_cand = A.shape
    counts = A.sum(axis=0)
    winner = int(np.argmax(counts))
    return DescriptiveResult(
        name="approval_vote",
        value=winner,
        extra={
            "counts": counts.tolist(),
            "n_voters": n_voters,
            "n_candidates": n_cand,
            "winner_approval_rate": float(counts[winner]) / n_voters,
        },
    )


appvl = approval_vote


def cheatsheet() -> str:
    return "approval_vote({}) -> Approval voting method."
