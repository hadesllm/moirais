# moirais.fn — function file (hadesllm/moirais)
"""Friedman two-way ANOVA by ranks (repeated measures)."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["frdmt"]


def frdmt(data, axis=0, cdf=None):
    r"""
    Friedman two-way ANOVA by ranks.

    Tests equality of k≥2 treatments for repeated measures design.
    Nonparametric alternative to two-way ANOVA.

    Parameters
    ----------
    data : array_like
        2D array: rows=blocks, columns=treatments.
    axis : int, optional
        Axis along which to rank (default 0).

    Returns
    -------
    dict
        Keys:
        - 'statistic': Friedman test statistic
        - 'p_value': p-value from chi-square distribution
        - 'k': number of treatments
        - 'b': number of blocks
        - 'interpretation': "reject" or "not reject" null
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("Input must be 2D array (blocks × treatments)")

    b = data.shape[0]  # blocks
    k = data.shape[1]  # treatments

    if b < 2 or k < 2:
        raise ValueError("Need at least 2 blocks and 2 treatments")

    # Rank within each block
    ranks = np.zeros_like(data)
    for i in range(b):
        ranks[i, :] = sp_stats.rankdata(data[i, :])

    # Sum of ranks per treatment
    R = np.sum(ranks, axis=0)

    # Friedman statistic
    T = 12 / (b * k * (k + 1)) * np.sum(R**2) - 3 * b * (k + 1)

    # Correction for ties
    ties_correction = 1
    for i in range(b):
        unique_vals = np.unique(data[i, :])
        for val in unique_vals:
            t = np.sum(data[i, :] == val)
            if t > 1:
                ties_correction -= t * (t - 1) * (t + 1) / (b * k * (k - 1) * (k + 1))

    if ties_correction > 0:
        T = T / ties_correction

    # p-value
    p_value = 1 - sp_stats.chi2.cdf(T, k - 1)

    return {
        "statistic": float(T),
        "p_value": float(p_value),
        "k": int(k),
        "b": int(b),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
