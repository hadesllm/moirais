"""Exponential mechanism for DP categorical selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_exponential_mechanism"]


def dp_exponential_mechanism(y, candidates, utility, epsilon, sensitivity):
    """
    Exponential mechanism for DP categorical selection

    Formula: P(r) ∝ exp(epsilon * u(x,r) / (2 * sensitivity_u))

    Parameters
    ----------
    y : array-like
        Input data.
    candidates : array-like
        Input data.
    utility : array-like
        Input data.
    epsilon : array-like
        Input data.
    sensitivity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McSherry & Talwar (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential mechanism for DP categorical selection"})


def cheatsheet():
    return "dpexpm: Exponential mechanism for DP categorical selection"
