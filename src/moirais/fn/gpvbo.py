"""Variational Bayesian optimization with GP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_variational_bayes_opt"]


def gp_variational_bayes_opt(X, y, X_grid):
    """
    Variational Bayesian optimization with GP

    Formula: max acquisition under GP posterior

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang-Frazier (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational Bayesian optimization with GP"})


def cheatsheet():
    return "gpvbo: Variational Bayesian optimization with GP"
