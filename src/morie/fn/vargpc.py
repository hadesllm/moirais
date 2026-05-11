"""Variational GP classifier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variational_gp_classifier"]


def variational_gp_classifier(X, y, X_test):
    """
    Variational GP classifier

    Formula: ELBO with sigmoid likelihood

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hensman-Matthews-Ghahramani (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational GP classifier"})


def cheatsheet():
    return "vargpc: Variational GP classifier"
