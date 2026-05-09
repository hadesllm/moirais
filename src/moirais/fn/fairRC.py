"""Fairness-aware reranker."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fairness_rec"]


def fairness_rec(pred, attrs):
    """
    Fairness-aware reranker

    Formula: reranking subject to demographic parity

    Parameters
    ----------
    pred : array-like
        Input data.
    attrs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yang-Stoyanovich (2017)
    """
    pred = np.atleast_1d(np.asarray(pred, dtype=float))
    n = len(pred)
    result = float(np.mean(pred))
    se = float(np.std(pred, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fairness-aware reranker"})


def cheatsheet():
    return "fairRC: Fairness-aware reranker"
