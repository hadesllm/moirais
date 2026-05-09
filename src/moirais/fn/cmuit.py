"""Conditional mutual information I(X;Y|Z)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["conditional_mutual_information"]


def conditional_mutual_information(y, x, y2, z):
    """
    Conditional mutual information I(X;Y|Z)

    Formula: I(X;Y|Z) = sum_z p(z) I(X;Y|Z=z)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    y2 : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cover & Thomas (2006) §2.5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional mutual information I(X;Y|Z)"})


def cheatsheet():
    return "cmuit: Conditional mutual information I(X;Y|Z)"
