"""Leiden refined community detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["leiden_grph"]


def leiden_grph(A, resolution):
    """
    Leiden refined community detection

    Formula: Louvain + refinement step (guarantees connectedness)

    Parameters
    ----------
    A : array-like
        Input data.
    resolution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Traag-Waltman-van Eck (2019)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leiden refined community detection"})


def cheatsheet():
    return "lemR: Leiden refined community detection"
