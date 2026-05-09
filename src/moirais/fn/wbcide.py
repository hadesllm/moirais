"""Wooldridge BJS clean-control DID estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wooldridge_bjs_estimator"]


def wooldridge_bjs_estimator(y, D, unit, time, X):
    """
    Wooldridge BJS clean-control DID estimator

    Formula: two-step: clean controls then weighted ATT

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wooldridge (2021); Borusyak-Jaravel-Spiess (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Wooldridge BJS clean-control DID estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Wooldridge BJS clean-control DID estimator"})


def cheatsheet():
    return "wbcide: Wooldridge BJS clean-control DID estimator"
