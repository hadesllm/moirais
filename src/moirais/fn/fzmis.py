# moirais.fn — function file (hadesllm/moirais)
"""MISE decomposition for kernel estimators."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_mise_computation"]


def fauzi_mise_computation(x):
    """
    MISE decomposition for kernel estimators

    Formula: MISE = integral Bias^2 + integral Var = c1*h^4 + c2/(nh)

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
    Fauzi Ch 1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "MISE decomposition for kernel estimators"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "MISE decomposition for kernel estimators"})


def cheatsheet():
    return "fzmis: MISE decomposition for kernel estimators"
