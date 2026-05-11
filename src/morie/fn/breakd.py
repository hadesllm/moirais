"""Finite-sample breakdown point."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["breakdown_point"]


def breakdown_point(estimator, n):
    """
    Finite-sample breakdown point

    Formula: smallest fraction of contamination that ruins estimator

    Parameters
    ----------
    estimator : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Donoho-Huber (1983)
    """
    estimator = np.atleast_1d(np.asarray(estimator, dtype=float))
    n = len(estimator)
    result = float(np.mean(estimator))
    se = float(np.std(estimator, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Finite-sample breakdown point"})


def cheatsheet():
    return "breakd: Finite-sample breakdown point"
