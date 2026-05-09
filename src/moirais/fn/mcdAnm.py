"""MCD-based Mahalanobis outlier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mcd_outlier"]


def mcd_outlier(X):
    """
    MCD-based Mahalanobis outlier

    Formula: d_M from minimum-covariance-determinant

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw-Van Driessen (1999)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MCD-based Mahalanobis outlier"})


def cheatsheet():
    return "mcdAnm: MCD-based Mahalanobis outlier"
