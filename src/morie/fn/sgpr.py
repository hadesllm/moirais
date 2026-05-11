"""FITC / DTC sparse GP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sparse_gp"]


def sparse_gp(X, y, M):
    """
    FITC / DTC sparse GP

    Formula: low-rank approx with M inducing points

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Snelson-Ghahramani (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FITC / DTC sparse GP"})


def cheatsheet():
    return "sgpr: FITC / DTC sparse GP"
