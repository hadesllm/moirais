"""Serendipity score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["serendipity"]


def serendipity(pred, baseline, relevant):
    """
    Serendipity score

    Formula: unexpected ∩ relevant

    Parameters
    ----------
    pred : array-like
        Input data.
    baseline : array-like
        Input data.
    relevant : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Adamopoulos-Tuzhilin (2014)
    """
    pred = np.atleast_1d(np.asarray(pred, dtype=float))
    n = len(pred)
    result = float(np.mean(pred))
    se = float(np.std(pred, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Serendipity score"})


def cheatsheet():
    return "servR: Serendipity score"
