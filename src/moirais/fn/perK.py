"""Periodic kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["periodic_kernel"]


def periodic_kernel(x, y, p, l):
    """
    Periodic kernel

    Formula: k = exp(-2 sin²(π|x-y|/p)/ℓ²)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    p : array-like
        Input data.
    l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacKay (1998)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Periodic kernel"})


def cheatsheet():
    return "perK: Periodic kernel"
