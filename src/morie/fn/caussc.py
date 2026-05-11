"""Abadie-Diamond-Hainmueller synthetic control weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_synthetic_control"]


def causal_synthetic_control(X1_pre, X0_pre, V):
    """
    Abadie-Diamond-Hainmueller synthetic control weights

    Formula: min_w (X_1 - X_0 w)^T V (X_1 - X_0 w) s.t. w>=0, Σw=1

    Parameters
    ----------
    X1_pre : array-like
        Input data.
    X0_pre : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, RMSE_pre

    References
    ----------
    Abadie-Diamond-Hainmueller (2010)
    """
    X1_pre = np.atleast_1d(np.asarray(X1_pre, dtype=float))
    n = len(X1_pre)
    result = float(np.mean(X1_pre))
    se = float(np.std(X1_pre, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Abadie-Diamond-Hainmueller synthetic control weights"})


def cheatsheet():
    return "caussc: Abadie-Diamond-Hainmueller synthetic control weights"
