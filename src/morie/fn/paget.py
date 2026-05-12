# morie.fn -- function file (hadesllm/morie)
"""Page's test for ordered alternatives (repeated measures)."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["paget"]


def paget(data, axis=0, cdf=None):
    r"""
    Page's test for ordered alternatives in repeated measures design.

    Tests H0: k treatments have equal effect vs
    H1: treatments have linear trend (e.g., μ₁ ≤ μ₂ ≤ ... ≤ μₖ).
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("Input must be 2D (blocks × treatments)")

    b = data.shape[0]  # blocks
    k = data.shape[1]  # treatments

    if b < 2 or k < 2:
        raise ValueError("Need ≥2 blocks and ≥2 treatments")

    # Rank within each block
    ranks = np.zeros_like(data)
    for i in range(b):
        ranks[i, :] = sp_stats.rankdata(data[i, :])

    # Sum of ranks per treatment
    R = np.sum(ranks, axis=0)

    # Page's L statistic: L = Σ j * R_j
    j_vals = np.arange(1, k + 1)
    L = np.sum(j_vals * R)

    # Expected value and variance
    E_L = b * k * (k + 1)**2 / 4
    Var_L = b * k * (k - 1) * (k + 1)**2 / 12

    # Standardized statistic
    z_stat = (L - E_L) / np.sqrt(Var_L)
    p_value = 1 - sp_stats.norm.cdf(z_stat)

    return {
        "statistic": float(L),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "k": int(k),
        "b": int(b),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
