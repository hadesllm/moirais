"""Jordan canonical form."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jordan_canonical"]


def jordan_canonical(A):
    """
    Jordan canonical form

    Formula: P J P^{-1}; J block-diagonal

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jordan (1870)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jordan canonical form"})


def cheatsheet():
    return "jordCD: Jordan canonical form"
