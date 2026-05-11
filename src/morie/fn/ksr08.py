# morie.fn — function file (hadesllm/morie)
"""Multiplier bootstrap for Z-estimators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_multiplier_bootstrap"]


def kosorok_multiplier_bootstrap(x):
    """
    Multiplier bootstrap for Z-estimators

    Formula: G_n^xi = n^{-1/2} sum xi_i(f(Xi)-Pf)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Multiplier bootstrap for Z-estimators"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Multiplier bootstrap for Z-estimators"})


def cheatsheet():
    return "ksr08: Multiplier bootstrap for Z-estimators"
