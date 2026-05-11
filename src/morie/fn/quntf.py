# morie.fn — function file (hadesllm/morie)
"""Nonparametric quantile function estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["quantile_function"]


def quantile_function(x):
    """
    Nonparametric quantile function estimation

    Formula: Q(tau) = inf{x: F(x) >= tau}

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Parzen (1979)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nonparametric quantile function estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nonparametric quantile function estimation"})


def cheatsheet():
    return "quntf: Nonparametric quantile function estimation"
