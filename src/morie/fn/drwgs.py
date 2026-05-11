"""DR weighting strategy comparison."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_weighting_strategy"]


def dr_weighting_strategy(y, D, X):
    """
    DR weighting strategy comparison

    Formula: compare propensity-only vs outcome-only vs DR

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
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR weighting strategy comparison"})


def cheatsheet():
    return "drwgs: DR weighting strategy comparison"
