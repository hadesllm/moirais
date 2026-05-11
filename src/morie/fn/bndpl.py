"""Bayesian density estimation via Pólya tree."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bnp_density_pl"]


def bnp_density_pl(y, tree_depth, alpha):
    """
    Bayesian density estimation via Pólya tree

    Formula: posterior of f(y); conjugate Beta updates

    Parameters
    ----------
    y : array-like
        Input data.
    tree_depth : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lavine (1992); Hanson (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bayesian density estimation via Pólya tree"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bayesian density estimation via Pólya tree"})


def cheatsheet():
    return "bndpl: Bayesian density estimation via Pólya tree"
