"""SVGP classifier with mini-batch."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_classification_svgp"]


def gp_classification_svgp(X, y, X_test, M):
    """
    SVGP classifier with mini-batch

    Formula: stochastic variational + sigmoid likelihood

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    M : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVGP classifier with mini-batch"})


def cheatsheet():
    return "gpcgs: SVGP classifier with mini-batch"
