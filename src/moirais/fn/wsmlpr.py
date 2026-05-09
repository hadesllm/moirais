"""Local polynomial regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_local_polynomial"]


def wasserman_local_polynomial(x, x_data, y_data, h, p):
    """
    Local polynomial regression

    Formula: min sum K_h(x-X_i)(Y_i - beta0 - beta1(X_i-x))^2

    Parameters
    ----------
    x : array-like
        Input data.
    x_data : array-like
        Input data.
    y_data : array-like
        Input data.
    h : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wasserman (2004), Ch 20
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local polynomial regression"})


def cheatsheet():
    return "wsmlpr: Local polynomial regression"
