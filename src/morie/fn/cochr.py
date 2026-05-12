# morie.fn -- function file (hadesllm/morie)
"""Cochran's Q test for k≥2 paired binary responses."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["cochr"]


def cochr(data, axis=0, cdf=None):
    r"""
    Cochran's Q test for k≥2 paired binary outcomes.

    Tests H0: k treatments have equal effect on binary response
    in a randomized block design.
    """
    data = np.asarray(data, dtype=int)

    if data.ndim != 2:
        raise ValueError("Input must be 2D (blocks × treatments)")

    b = data.shape[0]  # blocks
    k = data.shape[1]  # treatments

    if b < 2 or k < 2:
        raise ValueError("Need ≥2 blocks and ≥2 treatments")

    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must be binary (0/1)")

    # Column totals (treatment successes)
    G = np.sum(data, axis=0)

    # Row totals (block successes)
    L = np.sum(data, axis=1)

    # Cochran's Q statistic
    Q = (k * (k - 1) * np.sum(G**2) - (k * np.sum(L)**2)) / (k * np.sum(L) - np.sum(L**2))

    # p-value from chi-square with k-1 df
    p_value = 1 - sp_stats.chi2.cdf(Q, k - 1)

    return {
        "statistic": float(Q),
        "p_value": float(p_value),
        "k": int(k),
        "b": int(b),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
