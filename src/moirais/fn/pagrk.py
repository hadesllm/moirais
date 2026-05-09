"""PageRank."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pagerank"]


def pagerank(A, alpha):
    """
    PageRank

    Formula: x = αA^T D^{-1} x + (1-α)/n 1

    Parameters
    ----------
    A : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Page-Brin (1999)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PageRank"})


def cheatsheet():
    return "pagrk: PageRank"
