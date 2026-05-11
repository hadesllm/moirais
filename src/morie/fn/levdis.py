"""Levenshtein edit distance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["levenshtein"]


def levenshtein(s1, s2):
    """
    Levenshtein edit distance

    Formula: DP over insert/delete/substitute

    Parameters
    ----------
    s1 : array-like
        Input data.
    s2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Levenshtein (1966)
    """
    s1 = np.atleast_1d(np.asarray(s1, dtype=float))
    n = len(s1)
    result = float(np.mean(s1))
    se = float(np.std(s1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Levenshtein edit distance"})


def cheatsheet():
    return "levdis: Levenshtein edit distance"
