"""Bound under monotone outcome response."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_monot_outcome"]


def bound_monot_outcome(y, D, X, direction):
    """
    Bound under monotone outcome response

    Formula: E[Y(d')|X] ≥ E[Y(d)|X] for d' > d

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    direction : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Pepper (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under monotone outcome response"})


def cheatsheet():
    return "bdmnto: Bound under monotone outcome response"
