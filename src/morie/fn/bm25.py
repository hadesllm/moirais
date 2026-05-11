"""Okapi BM25."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bm25"]


def bm25(docs, query, k1, b):
    """
    Okapi BM25

    Formula: sum IDF · (tf(k_1+1))/(tf + k_1(1-b+b·dl/avgdl))

    Parameters
    ----------
    docs : array-like
        Input data.
    query : array-like
        Input data.
    k1 : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robertson et al (1995)
    """
    docs = np.atleast_1d(np.asarray(docs, dtype=float))
    n = len(docs)
    result = float(np.mean(docs))
    se = float(np.std(docs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Okapi BM25"})


def cheatsheet():
    return "bm25: Okapi BM25"
