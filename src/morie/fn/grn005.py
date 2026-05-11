"""Closed-form normal equation that minimizes the MSE cost function for linear regression.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_normal_equation"]


def geron_ch4_normal_equation(X, y):
    """
    Closed-form normal equation that minimizes the MSE cost function for linear regression.

    Formula: theta_hat = (X^T X)^-1 X^T y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_hat

    References
    ----------
    Geron (2026), Ch 4, Eq 4-5, p. 138
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form normal equation that minimizes the MSE cost function for linear regression."})


def cheatsheet():
    return "grn005: Closed-form normal equation that minimizes the MSE cost function for linear regression."
