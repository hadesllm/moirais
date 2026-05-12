# morie.fn -- function file (hadesllm/morie)
"""Sample entropy."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficult to see. Always in motion is the future."


def sample_entropy(x, m=2, r=0.2, **kwargs) -> DescriptiveResult:
    """Compute the sample entropy of signal *x*.

    Sample entropy quantifies signal regularity / complexity.
    Lower values indicate more regularity.

    Parameters
    ----------
    x : array-like
        Input signal.
    m : int
        Embedding dimension.
    r : float
        Tolerance (fraction of std, or absolute if std=0).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    sd = np.std(x, ddof=1)
    tol = r * sd if sd > 0 else r

    def _count_matches(dim):
        templates = np.array([x[i : i + dim] for i in range(N - dim)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < tol:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        se = float("inf")
    else:
        se = -float(np.log(A / B)) if A > 0 else float("inf")
    return DescriptiveResult(
        name="sample_entropy",
        value=se,
        extra={"sample_entropy": se, "m": m, "r": r, "tol": tol, "A": A, "B": B, "n": N},
    )


sentr = sample_entropy


def cheatsheet() -> str:
    return "sample_entropy({}) -> Sample entropy."
