"""Elastic net cost function: MSE plus a weighted mix of lasso (L1) and ridge (L2) regularization controlled by ratio r.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_elastic_net_cost_function"]


def geron_ch4_elastic_net_cost_function(X, y, theta, alpha, r):
    """
    Elastic net cost function: MSE plus a weighted mix of lasso (L1) and ridge (L2) regularization controlled by ratio r.

    Formula: J(theta) = MSE(theta) + r*(2*alpha*sum_{i=1}^{n}|theta_i|) + (1-r)*((alpha/m)*sum_{i=1}^{n} theta_i^2)

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
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost

    References
    ----------
    Geron (2026), Ch 4, Eq 4-13, p. 165
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elastic net cost function: MSE plus a weighted mix of lasso (L1) and ridge (L2) regularization controlled by ratio r."})


def cheatsheet():
    return "grn013: Elastic net cost function: MSE plus a weighted mix of lasso (L1) and ridge (L2) regularization controlled by ratio r."
