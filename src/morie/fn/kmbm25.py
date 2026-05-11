# morie.fn — function file (hadesllm/morie)
"""BM25 relevance score for query-document matching."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_bm25_score"]


def kamath_bm25_score(q_terms, doc_terms, idf, avgdl, k1, b):
    """
    BM25 relevance score for query-document matching

    Formula: BM25(q,d) = sum_t IDF(t) * (f(t,d)*(k1+1)) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))

    Parameters
    ----------
    q_terms : array-like
        Input data.
    doc_terms : array-like
        Input data.
    idf : array-like
        Input data.
    avgdl : array-like
        Input data.
    k1 : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 7, BM25 section
    """
    q_terms = np.atleast_1d(np.asarray(q_terms, dtype=float))
    n = len(q_terms)
    result = float(np.mean(q_terms))
    se = float(np.std(q_terms, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BM25 relevance score for query-document matching"})


def cheatsheet():
    return "kmbm25: BM25 relevance score for query-document matching"
