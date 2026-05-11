# morie.fn — function file (hadesllm/morie)
"""Mood's test for scale (dispersion)."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["moods"]


def moods(x, y, axis=0, cdf=None):
    r"""
    Mood's test for equality of scale parameters.

    Tests whether two independent samples have equal dispersion.
    Uses median scores based on combined median.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    if y.ndim == 2:
        y = np.take(y, 0, axis=axis)

    n_x = len(x)
    n_y = len(y)

    if n_x < 1 or n_y < 1:
        raise ValueError("Both samples must have ≥1 observation")

    combined = np.concatenate([x, y])
    med = np.median(combined)

    # Scores: (rank - (n+1)/2)² for each observation's position
    n = len(combined)
    indices = sp_stats.rankdata(combined)

    scores_x = []
    scores_y = []
    for i in range(n_x):
        rank_i = np.sum(combined <= x[i]) + np.sum((combined == x[i]) & (np.arange(n) <= np.where(combined == x[i])[0][0])) / 2
        scores_x.append((rank_i - (n + 1) / 2) ** 2)

    for i in range(n_y):
        rank_i = np.sum(combined <= y[i]) + np.sum((combined == y[i]) & (np.arange(n) <= np.where(combined == y[i])[0][0])) / 2
        scores_y.append((rank_i - (n + 1) / 2) ** 2)

    M_x = np.sum(scores_x)
    M_y = np.sum(scores_y)

    E_M = n_x * (n**2 - 1) / 12
    Var_M = n_x * n_y * (n**2 - 1) * (n**2 - 4) / (180 * (n - 1))

    z_stat = (M_x - E_M) / np.sqrt(Var_M)
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

    return {
        "statistic": float(M_x),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
