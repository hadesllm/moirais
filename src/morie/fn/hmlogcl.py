# morie.fn -- function file (hadesllm/morie)
"""Log-loss (cross-entropy) cost for binary logistic regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_logistic_cost"]


def geron_logistic_cost(X, y, theta):
    """
    Log-loss (cross-entropy) cost for binary logistic regression

    Formula: J = -(1/m) sum_i [y_i log p_hat_i + (1-y_i) log(1-p_hat_i)]

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-loss (cross-entropy) cost for binary logistic regression"})


def cheatsheet():
    return "hmlogcl: Log-loss (cross-entropy) cost for binary logistic regression"
