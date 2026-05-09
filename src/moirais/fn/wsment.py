"""Differential entropy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_entropy"]


def wasserman_entropy(p):
    """
    Differential entropy

    Formula: H(X) = -int p(x) log p(x) dx

    Parameters
    ----------
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 23
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differential entropy"})


def cheatsheet():
    return "wsment: Differential entropy"
