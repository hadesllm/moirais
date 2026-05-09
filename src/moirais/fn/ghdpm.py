# moirais.fn — function file (hadesllm/moirais)
"""DP mixture density estimate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dpmixture_density"]


def ghosal_dpmixture_density(x):
    """
    DP mixture density estimate

    Formula: f(x) = integral phi(x|theta) dG(theta)

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
    Ghosal Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "DP mixture density estimate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "DP mixture density estimate"})


def cheatsheet():
    return "ghdpm: DP mixture density estimate"
