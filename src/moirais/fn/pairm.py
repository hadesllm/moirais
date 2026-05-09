# moirais.fn — function file (hadesllm/moirais)
"""Pairwise comparison matrix from rankings."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pairwise_matrix(rankings) -> DescriptiveResult:
    """Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu"""
    R = np.asarray(rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("rankings must be 2D (voters x candidates).")
    n_voters, n_cand = R.shape
    M = np.zeros((n_cand, n_cand), dtype=int)
    for i in range(n_cand):
        for j in range(n_cand):
            if i != j:
                M[i, j] = int(np.sum(R[:, i] < R[:, j]))
    return DescriptiveResult(
        name="pairwise_matrix",
        value={"matrix": M},
        extra={"n_voters": n_voters, "n_candidates": n_cand},
    )


pairm = pairwise_matrix


def cheatsheet() -> str:
    return "pairwise_matrix({}) -> Build pairwise comparison matrix from rankings."
