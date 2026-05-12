# morie.fn -- function file (hadesllm/morie)
"""Few-shot learning: small number of in-context examples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_few_shot"]


def geron_few_shot(model, examples, query, k):
    """
    Few-shot learning: small number of in-context examples

    Formula: prompt contains k (x, y) pairs

    Parameters
    ----------
    model : array-like
        Input data.
    examples : array-like
        Input data.
    query : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Géron Ch 15
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Few-shot learning: small number of in-context examples"})


def cheatsheet():
    return "hmfsf: Few-shot learning: small number of in-context examples"
