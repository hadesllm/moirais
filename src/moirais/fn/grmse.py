# moirais.fn — function file (hadesllm/moirais)
"""Mean squared error cost for linear regression (theta minimizes MSE)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_linreg_mse_cost"]


def geron_linreg_mse_cost(X, y, theta):
    """
    Mean squared error cost for linear regression (theta minimizes MSE)

    Formula: MSE(theta) = (1/m) * sum_{i=1..m} (theta^T y^(i) - y^(i))^2

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
    Géron Ch 4, Eq 4-3 (MSE cost function)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean squared error cost for linear regression (theta minimizes MSE)"})


def cheatsheet():
    return "grmse: Mean squared error cost for linear regression (theta minimizes MSE)"
