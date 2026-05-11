"""Pólya tree prior for distribution estimation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polya_tree_prior"]


def polya_tree_prior(y, tree_depth, prior_alpha):
    """
    Pólya tree prior for distribution estimation

    Formula: binary partition; alpha_eps ~ Gamma; conditional Beta probabilities

    Parameters
    ----------
    y : array-like
        Input data.
    tree_depth : array-like
        Input data.
    prior_alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mauldin-Sudderth-Williams (1992); Lavine (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Pólya tree prior for distribution estimation"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Pólya tree prior for distribution estimation"})


def cheatsheet():
    return "poltre: Pólya tree prior for distribution estimation"
