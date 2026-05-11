# morie.fn — function file (hadesllm/morie)
"""Condorcet winner detection from pairwise preferences."""

from __future__ import annotations

from ._containers import DescriptiveResult


def condorcet_winner(preference_matrix) -> DescriptiveResult:
    """Out of chaos, comes order. — Friedrich Nietzsche"""
    import numpy as np

    M = np.asarray(preference_matrix, dtype=float)
    n = M.shape[0]
    winner = -1
    for i in range(n):
        beats_all = True
        for j in range(n):
            if i != j and M[i, j] <= M[j, i]:
                beats_all = False
                break
        if beats_all:
            winner = i
            break
    return DescriptiveResult(
        name="condorcet_winner",
        value=winner,
        extra={"n_candidates": n, "has_winner": winner >= 0},
    )


cndrc = condorcet_winner


def cheatsheet() -> str:
    return "condorcet_winner({}) -> Condorcet winner detection from pairwise preferences."
