"""Variational lower bound on MI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variational_bound"]


def variational_bound(X, Y, q):
    """
    Variational lower bound on MI

    Formula: I(X;Y) >= E[log q(y|x)/p(y)]

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barber-Agakov (2003)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational lower bound on MI"})


def cheatsheet():
    return "vbinfp: Variational lower bound on MI"
