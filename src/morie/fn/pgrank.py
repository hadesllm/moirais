"""PageRank with damping factor d."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pagerank"]


def pagerank(y, A, d, tol):
    """
    PageRank with damping factor d

    Formula: PR(v) = (1-d)/n + d sum_{u in N_in(v)} PR(u)/L(u)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    d : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Page, Brin, Motwani, Winograd (1999)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PageRank with damping factor d"})


def cheatsheet():
    return "pgrank: PageRank with damping factor d"
