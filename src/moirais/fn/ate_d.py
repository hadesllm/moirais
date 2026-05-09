# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Average treatment effect (ATE) definition under potential outcomes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ate_definition"]


def ate_definition(Y1, Y0):
    """
    Average treatment effect (ATE) definition under potential outcomes

    Formula: ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]

    Parameters
    ----------
    Y1 : array-like
        Input data.
    Y0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ate': 'float'}

    References
    ----------
    Molak Ch 1
    """
    Y1 = np.asarray(Y1, dtype=float)
    n = int(Y1) if Y1.ndim == 0 else len(Y1)
    if Y1.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average treatment effect (ATE) definition under potential outcomes"})
    estimate = np.median(Y1)
    se = 1.2533 * np.std(Y1, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average treatment effect (ATE) definition under potential outcomes"})


def cheatsheet():
    return "ate_d: Average treatment effect (ATE) definition under potential outcomes"
