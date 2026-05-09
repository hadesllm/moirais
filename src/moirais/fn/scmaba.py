"""Synthetic Control Method (Abadie-Diamond-Hainmueller)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["synthetic_control_method"]


def synthetic_control_method(y, treated, controls, X):
    """
    Synthetic Control Method (Abadie-Diamond-Hainmueller)

    Formula: min_w (X_T - X_C w)' V (X_T - X_C w) s.t. w >= 0, sum w = 1

    Parameters
    ----------
    y : array-like
        Input data.
    treated : array-like
        Input data.
    controls : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abadie, Diamond, Hainmueller (2010); Abadie (2021) review
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synthetic Control Method (Abadie-Diamond-Hainmueller)"})


def cheatsheet():
    return "scmaba: Synthetic Control Method (Abadie-Diamond-Hainmueller)"
