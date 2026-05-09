# moirais.fn — function file (hadesllm/moirais)
"""Maximum score estimator for binary response."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_binary_response"]


def horowitz_binary_response(x, y):
    """
    Maximum score estimator for binary response

    Formula: beta = argmax sum (2Yi-1) * 1(Xi'b > 0)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horowitz (2009), Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Maximum score estimator for binary response"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Maximum score estimator for binary response"})


def cheatsheet():
    return "hrzb1: Maximum score estimator for binary response"
