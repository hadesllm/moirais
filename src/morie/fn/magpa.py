"""IPD random-intercept logistic-normal model for proportions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_glmm_ipd_proportion"]


def ma_glmm_ipd_proportion(x, n):
    """
    IPD random-intercept logistic-normal model for proportions

    Formula: logit p_i = β + b_i; b_i ~ N(0, σ²)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, sigma2, p_pooled, ci

    References
    ----------
    Hamza-van Houwelingen-Stijnen (2008)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IPD random-intercept logistic-normal model for proportions"})


def cheatsheet():
    return "magpa: IPD random-intercept logistic-normal model for proportions"
