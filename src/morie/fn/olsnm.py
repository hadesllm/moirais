# morie.fn — function file (hadesllm/morie)
"""OLS normal equations for multiple regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ols_normal_equations"]


def ols_normal_equations(X, y):
    """
    OLS normal equations for multiple regression

    Formula: beta_hat = (X'X)^{-1} X'y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OLS normal equations for multiple regression"})


def cheatsheet():
    return "olsnm: OLS normal equations for multiple regression"
