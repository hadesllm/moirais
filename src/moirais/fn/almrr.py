# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Mean Reciprocal Rank over Q queries."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_mean_reciprocal_rank"]


def alammar_mean_reciprocal_rank(rankings, relevant_indices):
    """
    Mean Reciprocal Rank over Q queries

    Formula: MRR = (1/|Q|) sum_{q in Q} 1 / rank_of_first_relevant(q)

    Parameters
    ----------
    rankings : array-like
        Input data.
    relevant_indices : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mrr

    References
    ----------
    Alammar Ch 8, MRR section
    """
    rankings = np.atleast_1d(np.asarray(rankings, dtype=float))
    n = len(rankings)
    result = float(np.mean(rankings))
    se = float(np.std(rankings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean Reciprocal Rank over Q queries"})


def cheatsheet():
    return "almrr: Mean Reciprocal Rank over Q queries"
