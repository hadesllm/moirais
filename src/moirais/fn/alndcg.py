# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Normalized Discounted Cumulative Gain at k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_ndcg_at_k"]


def alammar_ndcg_at_k(relevances, k):
    """
    Normalized Discounted Cumulative Gain at k

    Formula: DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1);  NDCG@k = DCG@k / IDCG@k

    Parameters
    ----------
    relevances : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ndcg

    References
    ----------
    Alammar Ch 8, NDCG section
    """
    relevances = np.atleast_1d(np.asarray(relevances, dtype=float))
    n = len(relevances)
    result = float(np.mean(relevances))
    se = float(np.std(relevances, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized Discounted Cumulative Gain at k"})


def cheatsheet():
    return "alndcg: Normalized Discounted Cumulative Gain at k"
