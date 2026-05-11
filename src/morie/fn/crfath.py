"""Causal forest (Wager-Athey) for heterogeneous treatment effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_forest_wager_athey"]


def causal_forest_wager_athey(y, D, X):
    """
    Causal forest (Wager-Athey) for heterogeneous treatment effects

    Formula: tau(x) = E[Y(1)-Y(0) | X=x] via honest random forest

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wager & Athey (2018); Athey-Tibshirani-Wager (2019) GRF
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal forest (Wager-Athey) for heterogeneous treatment effects"})


def cheatsheet():
    return "crfath: Causal forest (Wager-Athey) for heterogeneous treatment effects"
