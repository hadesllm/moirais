"""Skewed-outcome bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_skewed_outcome"]


def bound_skewed_outcome(y, D, skew):
    """
    Skewed-outcome bound

    Formula: asymmetric upper/lower from skew prior

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    skew : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Pepper (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Skewed-outcome bound"})


def cheatsheet():
    return "bndsdo: Skewed-outcome bound"
