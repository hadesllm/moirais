# morie.fn -- function file (hadesllm/morie)
"""Copeland method for social choice."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def copeland_method(preference_matrix) -> DescriptiveResult:
    """An unexamined life is not worth living. -- Socrates"""
    M = np.asarray(preference_matrix, dtype=float)
    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError("preference_matrix must be square.")
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                if M[i, j] > M[j, i]:
                    scores[i] += 1
                elif M[i, j] < M[j, i]:
                    scores[i] -= 1
    winner = int(np.argmax(scores))
    return DescriptiveResult(
        name="copeland_method",
        value=winner,
        extra={"scores": scores.tolist(), "n_candidates": n},
    )


copld = copeland_method


def cheatsheet() -> str:
    return "copeland_method({}) -> Copeland pairwise comparison voting."
