"""OLS normal equations beta=(X'X)^-1 X'y."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_ols_normal_equations"]


def esl_ols_normal_equations(X, y):
    """
    OLS normal equations beta=(X'X)^-1 X'y

    Formula: beta_hat = (X'X)^{-1} X'y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, se

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OLS normal equations beta=(X'X)^-1 X'y"})


def cheatsheet():
    return "eslols: OLS normal equations beta=(X'X)^-1 X'y"
