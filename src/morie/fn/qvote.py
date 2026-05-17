# morie.fn -- function file (hadesllm/morie)
"""Quadratic voting model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def quadratic_voting(intensities, budget: float = 100.0) -> DescriptiveResult:
    """Quadratic voting: cost of k votes = k^2.

    Each voter allocates a budget across issues. Optimal vote allocation
    is proportional to the square root of preference intensity.

    :param intensities: (n_voters x n_issues) preference intensity matrix.
    :param budget: Per-voter voice credit budget.
    :return: DescriptiveResult with optimal vote allocations and totals.

    References
    ----------
    Armstrong (2014), Ch 4. Lalley & Weyl (2018).

    .. epigraph:: Give me a place to stand and I will move the earth. -- Archimedes
    """
    I = np.asarray(intensities, dtype=float)
    if I.ndim != 2:
        raise ValueError("intensities must be 2D (voters x issues).")
    n_voters, n_issues = I.shape

    signs = np.sign(I)
    abs_I = np.abs(I)
    sqrt_I = np.sqrt(abs_I)
    row_sums = sqrt_I.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    raw_votes = signs * np.sqrt(budget) * sqrt_I / row_sums
    totals = raw_votes.sum(axis=0)
    outcomes = (totals > 0).astype(int)

    return DescriptiveResult(
        name="quadratic_voting",
        value={"outcomes": outcomes, "totals": totals},
        extra={"votes": raw_votes, "budget": budget, "n_voters": n_voters, "n_issues": n_issues},
    )


qvote = quadratic_voting


def cheatsheet() -> str:
    return "quadratic_voting({}) -> Quadratic voting model."
