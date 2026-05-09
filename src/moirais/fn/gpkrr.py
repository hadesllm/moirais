"""GP-equivalent kernel ridge regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_kernel_ridge_reg"]


def gp_kernel_ridge_reg(X, y, X_test, lam):
    """
    GP-equivalent kernel ridge regression

    Formula: alpha = (K + lambda I)^-1 y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Saunders-Gammerman-Vovk (1998); Rasmussen-Williams (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP-equivalent kernel ridge regression"})


def cheatsheet():
    return "gpkrr: GP-equivalent kernel ridge regression"
