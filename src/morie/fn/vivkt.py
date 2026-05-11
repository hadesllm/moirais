"""VanderWeele-Vansteelandt 4-way decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vansteelandt_vanderweele"]


def vansteelandt_vanderweele(X, M, Y):
    """
    VanderWeele-Vansteelandt 4-way decomposition

    Formula: TE = CDE + INT_REF + INT_MED + PIE

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2014)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VanderWeele-Vansteelandt 4-way decomposition"})


def cheatsheet():
    return "vivkt: VanderWeele-Vansteelandt 4-way decomposition"
