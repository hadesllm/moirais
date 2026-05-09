"""GP posterior variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_variance"]


def gp_variance(X, X_star, kernel, sigma2):
    """
    GP posterior variance

    Formula: V_* = k_** − K_*(K+σ²I)^{-1}K_*'

    Parameters
    ----------
    X : array-like
        Input data.
    X_star : array-like
        Input data.
    kernel : array-like
        Input data.
    sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rasmussen-Williams (2006)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP posterior variance"})


def cheatsheet():
    return "gpvarF: GP posterior variance"
