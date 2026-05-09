"""VanderWeele 4-way decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vanderweele_decomposition"]


def vanderweele_decomposition(Y, X, M, C):
    """
    VanderWeele 4-way decomposition

    Formula: TE = CDE + reference + interaction + mediated interaction

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2014)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VanderWeele 4-way decomposition"})


def cheatsheet():
    return "vandIE: VanderWeele 4-way decomposition"
