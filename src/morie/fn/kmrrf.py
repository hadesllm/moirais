# morie.fn — function file (hadesllm/morie)
"""Reciprocal Rank Fusion across multiple rankers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_reciprocal_rank_fusion"]


def kamath_reciprocal_rank_fusion(rankings, k):
    """
    Reciprocal Rank Fusion across multiple rankers

    Formula: RRF(d) = sum_r 1 / (k + rank_r(d))

    Parameters
    ----------
    rankings : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fused_scores

    References
    ----------
    Kamath Ch 7, Reciprocal Rank Fusion section
    """
    rankings = np.atleast_1d(np.asarray(rankings, dtype=float))
    n = len(rankings)
    result = float(np.mean(rankings))
    se = float(np.std(rankings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reciprocal Rank Fusion across multiple rankers"})


def cheatsheet():
    return "kmrrf: Reciprocal Rank Fusion across multiple rankers"
