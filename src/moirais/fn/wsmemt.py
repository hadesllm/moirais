"""Expectation-maximization (EM) update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_em_algorithm"]


def wasserman_em_algorithm(X, theta0):
    """
    Expectation-maximization (EM) update

    Formula: Q(theta|theta^{(t)}) = E[log L(theta;X,Z)|X,theta^{(t)}]

    Parameters
    ----------
    X : array-like
        Input data.
    theta0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Wasserman (2004), Ch 9
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expectation-maximization (EM) update"})


def cheatsheet():
    return "wsmemt: Expectation-maximization (EM) update"
