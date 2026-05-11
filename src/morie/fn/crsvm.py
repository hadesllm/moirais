# morie.fn — function file (hadesllm/morie)
"""Cramer-von Mises test for goodness of fit."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["crsvm"]


def crsvm(x, dist="norm", dist_params=None, axis=0, cdf=None):
    r"""
    Cramer-von Mises goodness-of-fit test.

    Similar to K-S but uses squared distance, more sensitive to
    distributional shape overall.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be ≥2")

    x_sorted = np.sort(x)

    if dist == "norm":
        if dist_params is None:
            loc = np.mean(x)
            scale = np.std(x, ddof=1)
        else:
            loc = dist_params.get("loc", np.mean(x))
            scale = dist_params.get("scale", np.std(x, ddof=1))
        F = sp_stats.norm(loc=loc, scale=scale).cdf
    else:
        raise ValueError(f"Distribution {dist} not yet supported")

    F_vals = F(x_sorted)
    i = np.arange(1, n + 1)

    # Cramer-von Mises statistic
    W_sq = (1 / (12 * n)) + np.sum((F_vals - (2 * i - 1) / (2 * n)) ** 2)

    # Critical value and p-value (approximate)
    critical = 0.461
    p_value = 1.0 if W_sq < critical else 0.0

    return {
        "statistic": float(W_sq),
        "p_value": float(p_value),
        "critical_value": float(critical),
        "interpretation": "reject" if W_sq > critical else "not reject",
    }
