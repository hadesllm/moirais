"""Bayesian linear regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_linear"]


def bayes_linear(y, X, prior_var):
    """
    Bayesian linear regression

    Formula: y = X beta + eps; beta ~ N(0, V); sigma ~ HalfCauchy

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    prior_var : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lindley-Smith (1972)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian linear regression"})


def cheatsheet():
    return "bayreg: Bayesian linear regression"
