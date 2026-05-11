# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Recall@k — fraction of relevant items in the top-k retrieved set."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_recall_at_k"]


def alammar_recall_at_k(retrieved, relevant, k):
    """
    Recall@k — fraction of relevant items in the top-k retrieved set

    Formula: Recall@k = |relevant ∩ top_k| / |relevant|

    Parameters
    ----------
    retrieved : array-like
        Input data.
    relevant : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: recall

    References
    ----------
    Alammar Ch 8, Recall@K section
    """
    retrieved = np.atleast_1d(np.asarray(retrieved, dtype=float))
    n = len(retrieved)
    result = float(np.mean(retrieved))
    se = float(np.std(retrieved, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recall@k — fraction of relevant items in the top-k retrieved set"})


def cheatsheet():
    return "alrck: Recall@k — fraction of relevant items in the top-k retrieved set"
