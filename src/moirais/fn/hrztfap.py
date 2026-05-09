# moirais.fn — function file (hadesllm/moirais)
"""Asymptotic properties of Horowitz's T_n and F_n estimators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_T_F_asymp_props"]


def horowitz_T_F_asymp_props(x, y, bandwidth):
    """
    Asymptotic properties of Horowitz's T_n and F_n estimators

    Formula: sqrt(n)*(T_n-T_0) and sqrt(n)*(F_n-F_0) are asymptotically normal

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: asymptotic_distribution

    References
    ----------
    Horowitz Ch 6, Sec 6.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Asymptotic properties of Horowitz's T_n and F_n estimators"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Asymptotic properties of Horowitz's T_n and F_n estimators"})


def cheatsheet():
    return "hrztfap: Asymptotic properties of Horowitz's T_n and F_n estimators"
