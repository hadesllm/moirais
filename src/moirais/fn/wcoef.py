"""Kendall's coefficient of concordance W."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["wcoef"]


def wcoef(data, axis=0, cdf=None):
    r"""
    Kendall's coefficient of concordance W.

    Measures agreement among k≥2 judges/raters ranking n≥2 objects.
    W ∈ [0, 1]: 0=no agreement, 1=perfect agreement.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("Input must be 2D (judges × objects)")

    k = data.shape[0]  # judges
    n = data.shape[1]  # objects

    if k < 2 or n < 2:
        raise ValueError("Need ≥2 judges and ≥2 objects")

    # Rank each judge's ratings
    ranks = np.zeros_like(data)
    for i in range(k):
        ranks[i, :] = sp_stats.rankdata(data[i, :])

    # Sum of ranks per object
    R = np.sum(ranks, axis=0)

    # Concordance coefficient W
    S = np.sum((R - R.mean()) ** 2)
    W = 12 * S / (k**2 * (n**3 - n))

    # Chi-square test
    chi2_stat = k * (n - 1) * W
    p_value = 1 - sp_stats.chi2.cdf(chi2_stat, n - 1)

    return {
        "concordance": float(W),
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "k": int(k),
        "n": int(n),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
