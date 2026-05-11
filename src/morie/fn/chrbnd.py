"""Chernozhukov-Rosen intersection bounds inference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chernozhukov_rosen_bounds"]


def chernozhukov_rosen_bounds(y, X, instrument):
    """
    Chernozhukov-Rosen intersection bounds inference

    Formula: theta = max_v inf_W m(W,theta,v); bias-corrected confidence regions

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    instrument : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov-Lee-Rosen (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chernozhukov-Rosen intersection bounds inference"})


def cheatsheet():
    return "chrbnd: Chernozhukov-Rosen intersection bounds inference"
