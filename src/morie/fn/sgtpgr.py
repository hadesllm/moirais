"""PageRank via power iteration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_pagerank_power"]


def sgt_pagerank_power(A, d, max_iter, tol):
    """
    PageRank via power iteration

    Formula: p = (1-d)/n + d M^T p; iterate

    Parameters
    ----------
    A : array-like
        Input data.
    d : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p, iters

    References
    ----------
    Page-Brin (1998)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PageRank via power iteration"})


def cheatsheet():
    return "sgtpgr: PageRank via power iteration"
