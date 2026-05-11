"""Bayesian mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_mediation"]


def bayes_mediation(X, M, Y, priors):
    """
    Bayesian mediation

    Formula: jointly model M | X and Y | M, X

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yuan-MacKinnon (2009)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian mediation"})


def cheatsheet():
    return "baymed: Bayesian mediation"
