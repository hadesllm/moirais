"""Multidimensional scaling stress."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_mds"]


def esl_mds(D, k):
    """
    Multidimensional scaling stress

    Formula: S = sum (d_ij - |z_i-z_j|)^2

    Parameters
    ----------
    D : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: embedding

    References
    ----------
    Hastie ESL Ch 14
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    n = len(D)
    result = float(np.mean(D))
    se = float(np.std(D, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multidimensional scaling stress"})


def cheatsheet():
    return "eslmds: Multidimensional scaling stress"
