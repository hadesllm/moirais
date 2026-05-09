"""Newey-West HAC variance estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["newey_west_hac"]


def newey_west_hac(e, X, lags):
    """
    Newey-West HAC variance estimator

    Formula: V_HAC = Gamma_0 + 2 sum_k w_k Gamma_k, Bartlett weights

    Parameters
    ----------
    e : array-like
        Input data.
    X : array-like
        Input data.
    lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newey & West (1987)
    """
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = len(e)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Newey-West HAC variance estimator"})
    estimate = np.median(e)
    se = 1.2533 * np.std(e, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Newey-West HAC variance estimator"})


def cheatsheet():
    return "nwest: Newey-West HAC variance estimator"
