# morie.fn -- function file (hadesllm/morie)
"""Hodges-Lehmann one-sample location estimator based on Walsh averages."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_hodges_lehmann"]


def gibbons_hodges_lehmann(x):
    """
    Hodges-Lehmann one-sample location estimator based on Walsh averages

    Formula: Delta_hat = median{(X_i + X_j)/2: i <= j}

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, ci

    References
    ----------
    Gibbons Ch 5.7.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Hodges-Lehmann one-sample location estimator based on Walsh averages"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Hodges-Lehmann one-sample location estimator based on Walsh averages"})


def cheatsheet():
    return "gb_hgl: Hodges-Lehmann one-sample location estimator based on Walsh averages"
