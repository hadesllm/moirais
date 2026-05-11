# morie.fn — function file (hadesllm/morie)
"""Softmax function normalizes class scores into probabilities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_softmax_function"]


def geron_softmax_function(scores):
    """
    Softmax function normalizes class scores into probabilities

    Formula: p_k = exp(s_k) / sum_j exp(s_j)

    Parameters
    ----------
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 4
    """
    scores = np.atleast_1d(np.asarray(scores, dtype=float))
    n = len(scores)
    result = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Softmax function normalizes class scores into probabilities"})


def cheatsheet():
    return "hmsftm: Softmax function normalizes class scores into probabilities"
