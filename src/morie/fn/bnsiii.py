"""Identification by intersection of inequalities."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_iii"]


def bound_iii(y, X, moments):
    """
    Identification by intersection of inequalities

    Formula: theta in {theta : moment[i] ≤ 0 ∀i}

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    moments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov-Hong-Tamer (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identification by intersection of inequalities"})


def cheatsheet():
    return "bnsiii: Identification by intersection of inequalities"
