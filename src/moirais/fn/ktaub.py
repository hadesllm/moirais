# moirais.fn — function file (hadesllm/moirais)
"""Kendall's tau-b correlation with ties correction."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["ktaub"]


def ktaub(x, y, axis=0, cdf=None):
    r"""
    Kendall's tau-b correlation coefficient (with ties correction).

    Nonparametric measure of rank association between two variables.
    Corrects for ties in both variables.

    Parameters
    ----------
    x, y : array_like
        Two variables of equal length.
    axis : int, optional
        Axis along which to apply (default 0).

    Returns
    -------
    dict
        Keys:
        - 'correlation': Kendall's τ-b coefficient [-1, 1]
        - 'p_value': significance test p-value
        - 'n': sample size
        - 'concordant': number of concordant pairs
        - 'discordant': number of discordant pairs
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    if y.ndim == 2:
        y = np.take(y, 0, axis=axis)

    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be ≥2")

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            x_diff = x[i] - x[j]
            y_diff = y[i] - y[j]

            if x_diff != 0 and y_diff != 0:
                if (x_diff > 0 and y_diff > 0) or (x_diff < 0 and y_diff < 0):
                    concordant += 1
                else:
                    discordant += 1

    # Count ties
    x_ranks = sp_stats.rankdata(x)
    y_ranks = sp_stats.rankdata(y)

    unique_x = np.unique(x)
    unique_y = np.unique(y)

    tx = sum(np.sum(x == val) * (np.sum(x == val) - 1) for val in unique_x) / 2
    ty = sum(np.sum(y == val) * (np.sum(y == val) - 1) for val in unique_y) / 2

    # Tau-b
    denom = np.sqrt((n * (n - 1) / 2 - tx) * (n * (n - 1) / 2 - ty))
    if denom > 0:
        tau_b = (concordant - discordant) / denom
    else:
        tau_b = 0.0

    # p-value (normal approximation)
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1))
    z_stat = tau_b / np.sqrt(var_tau)
    p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

    return {
        "correlation": float(tau_b),
        "p_value": float(p_value),
        "n": int(n),
        "concordant": int(concordant),
        "discordant": int(discordant),
    }
