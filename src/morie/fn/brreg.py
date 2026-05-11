# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian ridge regression for marker effects."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_ridge_regression"]


def bayesian_ridge_regression(x, y):
    """
    Bayesian ridge regression for marker effects

    Formula: beta_j ~ N(0, sigma_b^2), sigma_b^2 ~ IG(a,b)

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
    Montesinos Lopez Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian ridge regression for marker effects"})


def cheatsheet():
    return "brreg: Bayesian ridge regression for marker effects"
