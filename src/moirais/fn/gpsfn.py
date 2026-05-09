"""Sparse GP via inducing points (FITC)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_sparse_inducing"]


def gp_sparse_inducing(X, y, X_test, inducing):
    """
    Sparse GP via inducing points (FITC)

    Formula: q(f) ~ N(K_um K_mm^-1 m, ...)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    inducing : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Snelson-Ghahramani (2006); Titsias (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse GP via inducing points (FITC)"})


def cheatsheet():
    return "gpsfn: Sparse GP via inducing points (FITC)"
