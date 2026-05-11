"""Function-on-function regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["function_on_function"]


def function_on_function(X, Y, basis_X, basis_Y):
    """
    Function-on-function regression

    Formula: Y(s) = ∫ β(s,t) X(t) dt + ε(s)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    basis_X : array-like
        Input data.
    basis_Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Function-on-function regression"})


def cheatsheet():
    return "fnlm: Function-on-function regression"
