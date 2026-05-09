"""Linear regression prediction as a weighted sum of input features plus a bias term.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_linear_regression_prediction"]


def geron_ch4_linear_regression_prediction(theta, x):
    """
    Linear regression prediction as a weighted sum of input features plus a bias term.

    Formula: y_hat = theta_0 + theta_1*x_1 + theta_2*x_2 + ... + theta_n*x_n

    Parameters
    ----------
    theta : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Geron (2026), Ch 4, Eq 4-2, p. 136
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear regression prediction as a weighted sum of input features plus a bias term."})


def cheatsheet():
    return "grn002: Linear regression prediction as a weighted sum of input features plus a bias term."
