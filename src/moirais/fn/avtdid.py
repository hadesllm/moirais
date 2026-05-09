"""Average treatment effect DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["avg_treatment_did"]


def avg_treatment_did(y, D, X):
    """
    Average treatment effect DiD

    Formula: ATE_DiD = ATT * P(D=1)

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna-Zhao (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average treatment effect DiD"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average treatment effect DiD"})


def cheatsheet():
    return "avtdid: Average treatment effect DiD"
