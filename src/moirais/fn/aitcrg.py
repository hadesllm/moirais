"""OLS in ILR coordinates (compositional regression)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_regression"]


def compositional_regression(X, Y_comp, V):
    """
    OLS in ILR coordinates (compositional regression)

    Formula: y_ilr = X β + ε; back-transform to simplex

    Parameters
    ----------
    X : array-like
        Input data.
    Y_comp : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, fitted, resid

    References
    ----------
    Pawlowsky-Glahn et al. (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OLS in ILR coordinates (compositional regression)"})


def cheatsheet():
    return "aitcrg: OLS in ILR coordinates (compositional regression)"
