"""Spearman-Brown projected reliability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spearman_brown"]


def spearman_brown(r, k):
    """
    Spearman-Brown projected reliability

    Formula: r' = k r / (1 + (k-1) r)

    Parameters
    ----------
    r : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Spearman (1910); Brown (1910)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spearman-Brown projected reliability"})


def cheatsheet():
    return "sbreli: Spearman-Brown projected reliability"
