# morie.fn -- function file (hadesllm/morie)
"""Jonckheere-Terpstra test for ordered alternatives."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["jnckt"]


def jnckt(*samples, axis=0):
    r"""
    Jonckheere-Terpstra test for ordered alternatives.

    Tests H0: k samples have same distribution vs
    H1: samples have ordered median (e.g., μ₁ ≤ μ₂ ≤ ... ≤ μₖ).
    """
    if len(samples) < 2:
        raise ValueError("At least 2 samples required")

    samples = [np.asarray(s, dtype=np.float64).ravel() for s in samples]
    k = len(samples)

    # Compute J statistic: count concordant pairs
    J = 0
    for i in range(k):
        for j in range(i + 1, k):
            # Count pairs where x_i < x_j
            for x_val in samples[i]:
                for y_val in samples[j]:
                    if x_val < y_val:
                        J += 1

    # Expected value and variance
    n_total = sum(len(s) for s in samples)
    E_J = n_total * (n_total - 1) / 4

    # Variance (approximate)
    Var_J = n_total * (n_total - 1) * (2 * n_total + 5) / 72

    # Standardized statistic
    z_stat = (J - E_J) / np.sqrt(Var_J)
    p_value = 1 - sp_stats.norm.cdf(z_stat)

    return {
        "statistic": float(J),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "k": int(k),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
