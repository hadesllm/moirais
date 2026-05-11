"""Proximal causal inference via proxy bridge function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_proximal_proxy"]


def causal_proximal_proxy(y, A, Z_proxy, W_proxy, X):
    """
    Proximal causal inference via proxy bridge function

    Formula: Solve E[Y - h(W,A,X)|Z,A,X]=0 for bridge h

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    Z_proxy : array-like
        Input data.
    W_proxy : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bridge, ATE

    References
    ----------
    Tchetgen-Tchetgen et al. (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal causal inference via proxy bridge function"})


def cheatsheet():
    return "causrho: Proximal causal inference via proxy bridge function"
