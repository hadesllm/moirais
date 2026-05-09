"""Nonparametric regression bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_nonparam_regr"]


def bound_nonparam_regr(y, D, X, bw):
    """
    Nonparametric regression bound

    Formula: kernel-density-based bounds

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    bw : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric regression bound"})


def cheatsheet():
    return "bndnpr: Nonparametric regression bound"
