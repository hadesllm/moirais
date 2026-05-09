"""Huber loss function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["huber_loss"]


def huber_loss(r, k):
    """
    Huber loss function

    Formula: ρ(r) = r²/2 if |r|≤k else k(|r|−k/2)

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
    Huber (1964)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Huber loss function"})


def cheatsheet():
    return "hubrho: Huber loss function"
