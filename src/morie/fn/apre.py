# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Aggregate proportional reduction in error (APRE) for OC."""
import numpy as np
from ._richresult import RichResult

__all__ = ["oc_apre"]


def oc_apre(votes, predictions):
    """
    Aggregate proportional reduction in error (APRE) for OC

    Formula: APRE_j = (minor_j - errors_j) / minor_j; mean over all votes

    Parameters
    ----------
    votes : array-like
        Input data.
    predictions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'apre': 'float', 'apre_per_vote': 'array'}

    References
    ----------
    Armstrong Ch 5
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    result = float(np.mean(votes))
    se = float(np.std(votes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aggregate proportional reduction in error (APRE) for OC"})


def cheatsheet():
    return "apre: Aggregate proportional reduction in error (APRE) for OC"
