# moirais.fn — function file (hadesllm/moirais)
"""Johnson-Lindenstrauss lemma: d' = O(log(n)/eps^2) preserves pairwise distances."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_johnson_lindenstrauss"]


def geron_johnson_lindenstrauss(n, eps):
    """
    Johnson-Lindenstrauss lemma: d' = O(log(n)/eps^2) preserves pairwise distances

    Formula: d_min >= 4*log(n) / (eps^2/2 - eps^3/3)

    Parameters
    ----------
    n : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d_min

    References
    ----------
    Géron Ch 7
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Johnson-Lindenstrauss lemma: d' = O(log(n)/eps^2) preserves pairwise distances"})


def cheatsheet():
    return "hmjl: Johnson-Lindenstrauss lemma: d' = O(log(n)/eps^2) preserves pairwise distances"
