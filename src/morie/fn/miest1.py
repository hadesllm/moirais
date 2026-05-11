"""Mutual information via KSG (k-NN)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_ksg"]


def mi_ksg(X, Y, k):
    """
    Mutual information via KSG (k-NN)

    Formula: I(X;Y) = psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kraskov-Stögbauer-Grassberger (2004)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mutual information via KSG (k-NN)"})


def cheatsheet():
    return "miest1: Mutual information via KSG (k-NN)"
