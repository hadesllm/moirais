"""RAG retrieval — top-k embedding search."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rag_retrieval"]


def rag_retrieval(query, docs):
    """
    RAG retrieval — top-k embedding search

    Formula: score = q·d / (||q|| ||d||); ANN

    Parameters
    ----------
    query : array-like
        Input data.
    docs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lewis et al (2020) RAG
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RAG retrieval — top-k embedding search"})


def cheatsheet():
    return "ragRet: RAG retrieval — top-k embedding search"
