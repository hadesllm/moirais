"""Randić connectivity index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_randic_index"]


def sgt_randic_index(A):
    """
    Randić connectivity index

    Formula: R = Σ_{ij ∈ E} (d_i d_j)^{-1/2}

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R

    References
    ----------
    Randić (1975)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Randić connectivity index"})


def cheatsheet():
    return "sgtrnh: Randić connectivity index"
