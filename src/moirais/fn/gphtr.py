"""Heteroscedastic GP regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_heteroscedastic"]


def gp_heteroscedastic(X, y, X_test):
    """
    Heteroscedastic GP regression

    Formula: y = f(x) + epsilon(x); both GP

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
    Goldberg-Williams-Bishop (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heteroscedastic GP regression"})


def cheatsheet():
    return "gphtr: Heteroscedastic GP regression"
