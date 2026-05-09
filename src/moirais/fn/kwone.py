# moirais.fn — function file (hadesllm/moirais)
"""Kruskal-Wallis one-way ANOVA by ranks."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["kwone"]


def kwone(*samples, axis=0):
    r"""
    Kruskal-Wallis one-way ANOVA (nonparametric).

    Tests equality of k≥2 independent sample distributions.
    Generalization of Mann-Whitney to k groups.
    """
    if len(samples) < 2:
        raise ValueError("At least 2 samples required")

    samples = [np.asarray(s, dtype=np.float64).ravel() for s in samples]
    combined = np.concatenate(samples)
    n_total = len(combined)

    if n_total < len(samples) * 2:
        raise ValueError("Insufficient total observations")

    ranks = sp_stats.rankdata(combined)

    # H statistic
    H = 0
    rank_idx = 0
    for sample in samples:
        n_i = len(sample)
        R_i = np.sum(ranks[rank_idx : rank_idx + n_i])
        H += R_i**2 / n_i
        rank_idx += n_i

    H = 12 / (n_total * (n_total + 1)) * H - 3 * (n_total + 1)

    # Correction for ties
    unique_vals = np.unique(combined)
    ties_correction = 1
    for val in unique_vals:
        t = np.sum(combined == val)
        if t > 1:
            ties_correction -= t * (t - 1) * (t + 1) / (n_total * (n_total - 1) * (n_total + 1))

    if ties_correction > 0:
        H = H / ties_correction

    # p-value from chi-square with k-1 df
    k = len(samples)
    p_value = 1 - sp_stats.chi2.cdf(H, k - 1)

    return {
        "statistic": float(H),
        "p_value": float(p_value),
        "k": int(k),
        "n_total": int(n_total),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
