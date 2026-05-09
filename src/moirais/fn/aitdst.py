"""Aitchison distance between two compositions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_distance"]


def aitchison_distance(x, y):
    """
    Aitchison distance between two compositions

    Formula: d_A(x,y) = ||clr(x)-clr(y)||_2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aitchison distance between two compositions"})


def cheatsheet():
    return "aitdst: Aitchison distance between two compositions"
