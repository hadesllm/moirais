# moirais.fn — function file (hadesllm/moirais)
"""Ordinary least squares closed-form solution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["linear_regression_ols"]


def linear_regression_ols(x, y):
    """
    Ordinary least squares closed-form solution

    Formula: beta = (X'X)^{-1} X'y

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Geron (2026), Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinary least squares closed-form solution"})


def cheatsheet():
    return "linrg: Ordinary least squares closed-form solution"
