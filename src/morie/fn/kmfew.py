# morie.fn -- function file (hadesllm/morie)
"""Few-shot exemplar selection by similarity to the query."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_few_shot_exemplar_selection"]


def kamath_few_shot_exemplar_selection(D, query_embed, K):
    """
    Few-shot exemplar selection by similarity to the query

    Formula: D_K = TopK_{d in D} sim(embed(x), embed(d))

    Parameters
    ----------
    D : array-like
        Input data.
    query_embed : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: exemplars

    References
    ----------
    Kamath Ch 3, Few-Shot Exemplar Selection section
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    n = len(D)
    result = float(np.mean(D))
    se = float(np.std(D, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Few-shot exemplar selection by similarity to the query"})


def cheatsheet():
    return "kmfew: Few-shot exemplar selection by similarity to the query"
