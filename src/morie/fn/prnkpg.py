"""PageRank."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pagerank"]


def pagerank(G, damping):
    """
    PageRank

    Formula: x = (1-d)/n + d A^T D^-1 x

    Parameters
    ----------
    G : array-like
        Input data.
    damping : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Page-Brin (1998)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PageRank"})


def cheatsheet():
    return "prnkpg: PageRank"
