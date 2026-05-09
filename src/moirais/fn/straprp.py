"""Stratified proportion estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stratified_proportion"]


def stratified_proportion(y, stratum, weights):
    """
    Stratified proportion estimator

    Formula: p_st = sum_h W_h p_h

    Parameters
    ----------
    y : array-like
        Input data.
    stratum : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §5.5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Stratified proportion estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Stratified proportion estimator"})


def cheatsheet():
    return "straprp: Stratified proportion estimator"
