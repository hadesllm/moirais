"""Surprisal -log p(x)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["surprisal"]


def surprisal(p, x):
    """
    Surprisal -log p(x)

    Formula: I(x) = -log p(x)

    Parameters
    ----------
    p : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shannon (1948)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Surprisal -log p(x)"})


def cheatsheet():
    return "surprl: Surprisal -log p(x)"
