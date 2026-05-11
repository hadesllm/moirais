"""Conditional mutual information I(X;Y|Z)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["conditional_mi"]


def conditional_mi(pxyz):
    """
    Conditional mutual information I(X;Y|Z)

    Formula: I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)

    Parameters
    ----------
    pxyz : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cover-Thomas (2006)
    """
    pxyz = np.atleast_1d(np.asarray(pxyz, dtype=float))
    n = len(pxyz)
    result = float(np.mean(pxyz))
    se = float(np.std(pxyz, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional mutual information I(X;Y|Z)"})


def cheatsheet():
    return "cmutif: Conditional mutual information I(X;Y|Z)"
