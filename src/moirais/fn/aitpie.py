"""Pielou evenness index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_pielou"]


def compositional_pielou(x):
    """
    Pielou evenness index

    Formula: J = H/log(D)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: J

    References
    ----------
    Pielou (1966)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pielou evenness index"})


def cheatsheet():
    return "aitpie: Pielou evenness index"
