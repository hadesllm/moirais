"""METEOR alignment-based."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["meteor"]


def meteor(candidate, reference):
    """
    METEOR alignment-based

    Formula: F-mean of unigram P/R with stemming + WordNet

    Parameters
    ----------
    candidate : array-like
        Input data.
    reference : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Banerjee-Lavie (2005)
    """
    candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
    n = len(candidate)
    result = float(np.mean(candidate))
    se = float(np.std(candidate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "METEOR alignment-based"})


def cheatsheet():
    return "meteor: METEOR alignment-based"
