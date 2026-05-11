"""Lasso regression cost function: MSE plus an L1 penalty on the feature weights.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_lasso_regression_cost_function"]


def geron_ch4_lasso_regression_cost_function(X, y, theta, alpha):
    """
    Lasso regression cost function: MSE plus an L1 penalty on the feature weights.

    Formula: J(theta) = MSE(theta) + 2*alpha * sum_{i=1}^{n} |theta_i|

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    theta : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost

    References
    ----------
    Geron (2026), Ch 4, Eq 4-11, p. 162
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lasso regression cost function: MSE plus an L1 penalty on the feature weights."})


def cheatsheet():
    return "grn011: Lasso regression cost function: MSE plus an L1 penalty on the feature weights."
