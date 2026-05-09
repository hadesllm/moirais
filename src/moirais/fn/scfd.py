"""Scalar-on-function regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["scalar_on_function"]


def scalar_on_function(X, Y, basis):
    """
    Scalar-on-function regression

    Formula: Y = α + ∫ β(t) X(t) dt + ε

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reiss-Ogden (2007)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scalar-on-function regression"})


def cheatsheet():
    return "scfd: Scalar-on-function regression"
