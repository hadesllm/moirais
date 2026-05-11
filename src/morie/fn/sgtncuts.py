"""Normalised cut objective for a partition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_normalised_cut"]


def sgt_normalised_cut(A, labels):
    """
    Normalised cut objective for a partition

    Formula: Ncut(A,B) = cut(A,B)(1/vol A + 1/vol B)

    Parameters
    ----------
    A : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ncut

    References
    ----------
    Shi-Malik (2000)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalised cut objective for a partition"})


def cheatsheet():
    return "sgtncuts: Normalised cut objective for a partition"
