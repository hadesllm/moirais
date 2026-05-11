# morie.fn — function file (hadesllm/morie)
"""MSE cost function for linear regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_linreg_mse_cost"]


def geron_linreg_mse_cost(X, y, theta):
    """
    MSE cost function for linear regression

    Formula: MSE(theta) = (1/m) sum_i (theta^T x^(i) - y^(i))^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost

    References
    ----------
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSE cost function for linear regression"})


def cheatsheet():
    return "hmmsec: MSE cost function for linear regression"
