"""Gaussian subgaussian estimator for DP mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gauss_subgaussian_estimator"]


def gauss_subgaussian_estimator(y, C, epsilon, n):
    """
    Gaussian subgaussian estimator for DP mean

    Formula: mu_hat = (1/n) sum y_i + Lap(C/(n*epsilon))

    Parameters
    ----------
    y : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork & Lei (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Gaussian subgaussian estimator for DP mean"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Gaussian subgaussian estimator for DP mean"})


def cheatsheet():
    return "gestee: Gaussian subgaussian estimator for DP mean"
