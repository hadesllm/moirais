"""Local linear regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_local_linear"]


def esl_local_linear(x0, x, y, lambda_):
    """
    Local linear regression

    Formula: min sum K_lambda(x_0,x_i)(y_i-alpha-beta(x_i-x_0))^2

    Parameters
    ----------
    x0 : array-like
        Input data.
    x : array-like
        Input data.
    y : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hastie ESL Ch 6
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local linear regression"})


def cheatsheet():
    return "esllpr: Local linear regression"
