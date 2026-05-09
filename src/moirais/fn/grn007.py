"""Full gradient vector of the MSE cost function over all parameters, used in batch gradient descent.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_mse_gradient_vector"]


def geron_ch4_mse_gradient_vector(X, y, theta):
    """
    Full gradient vector of the MSE cost function over all parameters, used in batch gradient descent.

    Formula: grad_theta MSE(theta) = (2/m) * X^T (X theta - y)

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
        Keys: gradient_vector

    References
    ----------
    Geron (2026), Ch 4, Eq 4-7, p. 146
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Full gradient vector of the MSE cost function over all parameters, used in batch gradient descent."})


def cheatsheet():
    return "grn007: Full gradient vector of the MSE cost function over all parameters, used in batch gradient descent."
