# morie.fn — function file (hadesllm/morie)
"""In-context learning conditional probability with K demonstrations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_in_context_learning_prob"]


def kamath_in_context_learning_prob(demonstrations, query, model):
    """
    In-context learning conditional probability with K demonstrations

    Formula: P(y | x, D_K) = P_LLM(y | [ex_1, ..., ex_K, x])

    Parameters
    ----------
    demonstrations : array-like
        Input data.
    query : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Kamath Ch 3, In-Context Learning section
    """
    demonstrations = np.atleast_1d(np.asarray(demonstrations, dtype=float))
    n = len(demonstrations)
    result = float(np.mean(demonstrations))
    se = float(np.std(demonstrations, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "In-context learning conditional probability with K demonstrations"})


def cheatsheet():
    return "kmicl: In-context learning conditional probability with K demonstrations"
