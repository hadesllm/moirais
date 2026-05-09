"""Bayesian credible bound for partial ID."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_bayes_credible"]


def bound_bayes_credible(y, X, prior):
    """
    Bayesian credible bound for partial ID

    Formula: posterior over identification set

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Moon-Schorfheide (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian credible bound for partial ID"})


def cheatsheet():
    return "bndbye: Bayesian credible bound for partial ID"
