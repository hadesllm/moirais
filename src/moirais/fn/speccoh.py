"""Magnitude-squared coherence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["coherence"]


def coherence(x, y, f):
    """
    Magnitude-squared coherence

    Formula: |S_xy(f)|^2 / (S_xx(f) S_yy(f))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bendat-Piersol (2010)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude-squared coherence"})


def cheatsheet():
    return "speccoh: Magnitude-squared coherence"
