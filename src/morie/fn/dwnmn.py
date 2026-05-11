# morie.fn — function file (hadesllm/morie)
"""Dynamic ideal point estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dynamic_wnominate"]


def dynamic_wnominate(x):
    """
    Dynamic ideal point estimation

    Formula: x_i,t* = x_i,t-1* + w_t (random walk)

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
    Armstrong Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Dynamic ideal point estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Dynamic ideal point estimation"})


def cheatsheet():
    return "dwnmn: Dynamic ideal point estimation"
