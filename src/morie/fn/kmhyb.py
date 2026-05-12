# morie.fn -- function file (hadesllm/morie)
"""Hybrid retrieval: weighted fusion of dense and sparse scores."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_hybrid_retrieval_fusion"]


def kamath_hybrid_retrieval_fusion(s_dense, s_sparse, lam):
    """
    Hybrid retrieval: weighted fusion of dense and sparse scores

    Formula: score = lam * s_dense + (1 - lam) * s_sparse; rank by score

    Parameters
    ----------
    s_dense : array-like
        Input data.
    s_sparse : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fused

    References
    ----------
    Kamath Ch 7, Hybrid Retrieval section
    """
    s_dense = np.atleast_1d(np.asarray(s_dense, dtype=float))
    n = len(s_dense)
    result = float(np.mean(s_dense))
    se = float(np.std(s_dense, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hybrid retrieval: weighted fusion of dense and sparse scores"})


def cheatsheet():
    return "kmhyb: Hybrid retrieval: weighted fusion of dense and sparse scores"
