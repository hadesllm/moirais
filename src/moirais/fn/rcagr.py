# moirais.fn — function file (hadesllm/moirais)
"""Pairwise agreement scores. 'Destructo Disc!' -- Krillin, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def agreement_scores(votes):
    """Compute pairwise agreement matrix from vote/preference matrix.

    Parameters
    ----------
    votes : array-like
        Binary vote matrix (n_voters x n_items). 1=yes, 0=no.

    Returns
    -------
    DescriptiveResult
        value = agreement matrix (n_voters x n_voters), values in [0, 1].
    """
    import numpy as np

    V = np.asarray(votes, dtype=float)
    n = V.shape[0]
    m = V.shape[1]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = float(np.sum(V[i] == V[j])) / m
    return DescriptiveResult(name="agreement_scores", value=A, extra={"n_voters": n, "n_items": m})


rcagr = agreement_scores


def cheatsheet() -> str:
    return "agreement_scores({}) -> Pairwise agreement scores. 'Destructo Disc!' -- Krillin, Dra"
