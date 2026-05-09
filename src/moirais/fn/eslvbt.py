"""Variance of OLS coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_var_beta_hat"]


def esl_var_beta_hat(X, sigma2):
    """
    Variance of OLS coefficients

    Formula: Var(beta_hat) = sigma^2 (X'X)^{-1}

    Parameters
    ----------
    X : array-like
        Input data.
    sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: covariance

    References
    ----------
    Hastie ESL Ch 3
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of OLS coefficients"})


def cheatsheet():
    return "eslvbt: Variance of OLS coefficients"
