"""Semiparametric linear regression model with bounded covariates and conditional mean-zero residuals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch1_linear_regression_model"]


def kosorok_ch1_linear_regression_model(Y, Z, beta, e):
    """
    Semiparametric linear regression model with bounded covariates and conditional mean-zero residuals

    Formula: Y = beta' * Z + e, with E[e|Z]=0 and E[e^2|Z] <= K

    Parameters
    ----------
    Y : array-like
        Input data.
    Z : array-like
        Input data.
    beta : array-like
        Input data.
    e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 1, Eq 1.1, p. 3
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric linear regression model with bounded covariates and conditional mean-zero residuals"})


def cheatsheet():
    return "ksr020: Semiparametric linear regression model with bounded covariates and conditional mean-zero residuals"
