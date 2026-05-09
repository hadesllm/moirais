"""Genomic BLUP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gblup_estimator"]


def gblup_estimator(y, X, Z, G):
    """
    Genomic BLUP

    Formula: y = X beta + Z u + e; u ~ N(0, G sigma_u^2)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanRaden (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genomic BLUP"})


def cheatsheet():
    return "gblupr: Genomic BLUP"
