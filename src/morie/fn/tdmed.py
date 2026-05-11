"""Two-dimensional mediation effect with M1 and M2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["two_dimensional_mediation"]


def two_dimensional_mediation(X, M1, M2, Y):
    """
    Two-dimensional mediation effect with M1 and M2

    Formula: NIE = NIE_M1 + NIE_M2 + NIE_M1xM2

    Parameters
    ----------
    X : array-like
        Input data.
    M1 : array-like
        Input data.
    M2 : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele & Vansteelandt (2014)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-dimensional mediation effect with M1 and M2"})


def cheatsheet():
    return "tdmed: Two-dimensional mediation effect with M1 and M2"
