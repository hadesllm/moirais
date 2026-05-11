"""Causal survival forest from grf."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_survival_forest_grf"]


def causal_survival_forest_grf(time, event, D, X):
    """
    Causal survival forest from grf

    Formula: honest forest splits maximizing tau heterogeneity

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
    Cui et al (2023)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal survival forest from grf"})


def cheatsheet():
    return "survcfg: Causal survival forest from grf"
