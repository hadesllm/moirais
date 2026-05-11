"""Novelty (popularity-inverse)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["novelty"]


def novelty(item, popularity):
    """
    Novelty (popularity-inverse)

    Formula: -log_2 P(item)

    Parameters
    ----------
    item : array-like
        Input data.
    popularity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vargas-Castells (2011)
    """
    item = np.atleast_1d(np.asarray(item, dtype=float))
    n = len(item)
    result = float(np.mean(item))
    se = float(np.std(item, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Novelty (popularity-inverse)"})


def cheatsheet():
    return "novlt: Novelty (popularity-inverse)"
