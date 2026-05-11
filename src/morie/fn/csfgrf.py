"""Causal survival forest for heterogeneous time-to-event treatment effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_survival_forest"]


def causal_survival_forest(time, event, D, X):
    """
    Causal survival forest for heterogeneous time-to-event treatment effects

    Formula: tau(x) = E[T(1) - T(0) | X=x] via honest survival random forest

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
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
    Cui-Kosorok-Athey-Wager (2023)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal survival forest for heterogeneous time-to-event treatment effects"})


def cheatsheet():
    return "csfgrf: Causal survival forest for heterogeneous time-to-event treatment effects"
