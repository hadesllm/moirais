"""Siegel-Tukey test for scale equality."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["stkey"]


def stkey(x, y, axis=0, cdf=None):
    r"""
    Siegel-Tukey test for equality of scale parameters.

    Alternative to FABD that uses "alternating" rank scores,
    more weight on extreme values.
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

    # Siegel-Tukey scores: alternating from outside in
    # 1, 2n, 3, 2n-2, 5, 2n-4, ...
    st_scores = np.zeros(n)
    low_val = 1
    high_val = n

    for i in range(n):
        if i % 2 == 0:
            st_scores[ranks.argsort()[i]] = low_val
            low_val += 1
        else:
            st_scores[ranks.argsort()[i]] = high_val
            high_val -= 1

    # Sum of ST scores for x
    T = np.sum(st_scores[:n_x])

    # Expected and variance
    E_T = n_x * (n + 1) / 2
    Var_T = (n_x * n_y * (n + 1)**2) / (12 * (n - 1))

    # Standardized statistic
    z_stat = (T - E_T) / np.sqrt(Var_T)
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

    return {
        "statistic": float(T),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }
