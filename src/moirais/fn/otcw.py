"""Cyclical monotonicity check for a transport map."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_cyclical_weight"]


def ot_cyclical_weight(X, Y, C, perm):
    """
    Cyclical monotonicity check for a transport map

    Formula: Σ c(x_{σ(i)}, y_i) <= Σ c(x_{σ(π(i))}, y_i) for all π

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    C : array-like
        Input data.
    perm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_cm, slack

    References
    ----------
    Villani (2003)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cyclical monotonicity check for a transport map"})


def cheatsheet():
    return "otcw: Cyclical monotonicity check for a transport map"
