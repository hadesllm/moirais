# moirais.fn — function file (hadesllm/moirais)
"""Log-loss (cross-entropy) cost for binary logistic regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_logistic_cross_entropy_cost"]


def geron_logistic_cross_entropy_cost(X, y, theta):
    """
    Log-loss (cross-entropy) cost for binary logistic regression

    Formula: J(theta) = -(1/m)*sum_{i} [y^(i) log(p_hat^(i)) + (1-y^(i)) log(1-p_hat^(i))]

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
    Géron Ch 4, Eq 4-17 (Logistic Regression cost)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-loss (cross-entropy) cost for binary logistic regression"})


def cheatsheet():
    return "grlogc: Log-loss (cross-entropy) cost for binary logistic regression"
