# moirais.fn — function file (hadesllm/moirais)
"""Klotz normal-scores test for scale equality."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["klotz"]


def klotz(x, y, axis=0, cdf=None):
    r"""
    Klotz test for equality of scale parameters using normal scores.

    Uses van der Waerden scores: Φ⁻¹(rank/(n+1)).
    More sensitive to extremes than FABD.
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
    n = len(combined)

    # Rank combined sample
    ranks = sp_stats.rankdata(combined)

    # Klotz scores (squared normal scores): (Φ⁻¹(rank/(n+1)))²
    van_der_waerden = sp_stats.norm.ppf(ranks / (n + 1))
    klotz_scores = van_der_waerden ** 2

    # Sum of squared normal scores for x
    T = np.sum(klotz_scores[:n_x])

    # Expected value
    E_T = n_x * (n + 1) / 4

    # Variance (approximation)
    Var_T = n_x * n_y * (n**2 - 1) / (12 * (n - 1))

    # Standardized statistic
    z_stat = (T - E_T) / np.sqrt(Var_T)
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

    return {
        "statistic": float(T),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
