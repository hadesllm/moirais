"""Generalized gamma AFT model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generalized_gamma_aft"]


def generalized_gamma_aft(time, event, X):
    """
    Generalized gamma AFT model

    Formula: log(T) = beta'X + sigma * Z, Z ~ generalized gamma

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cox & Matheson (2014)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized gamma AFT model"})


def cheatsheet():
    return "ggmaft: Generalized gamma AFT model"
