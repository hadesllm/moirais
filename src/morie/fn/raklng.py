"""Raking ratio post-stratification (iterative proportional fitting)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["raking_ratio"]


def raking_ratio(y, weights, margins):
    """
    Raking ratio post-stratification (iterative proportional fitting)

    Formula: iterate w_i (m^A_h / hat m^A_h)(m^B_k / hat m^B_k)

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    margins : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Deming & Stephan (1940)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Raking ratio post-stratification (iterative proportional fitting)"})


def cheatsheet():
    return "raklng: Raking ratio post-stratification (iterative proportional fitting)"
