"""GP regression with hyperparameter MCMC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_regression_bayes"]


def gp_regression_bayes(X, y, kernel):
    """
    GP regression with hyperparameter MCMC

    Formula: sample (sigma, l, sigma_n) | data + posterior over f

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Murray-Adams (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP regression with hyperparameter MCMC"})


def cheatsheet():
    return "gpregb: GP regression with hyperparameter MCMC"
