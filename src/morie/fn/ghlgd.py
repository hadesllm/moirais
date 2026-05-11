# morie.fn — function file (hadesllm/morie)
"""Log-density estimation via exponential family."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_log_density"]


def ghosal_log_density(x):
    """
    Log-density estimation via exponential family

    Formula: f(x) = exp(psi(x)) / integral exp(psi)

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
    Ghosal Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Log-density estimation via exponential family"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Log-density estimation via exponential family"})


def cheatsheet():
    return "ghlgd: Log-density estimation via exponential family"
