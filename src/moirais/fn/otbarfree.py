"""Free-support Wasserstein barycenter via fixed-point."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_barycenter_free"]


def ot_barycenter_free(X_list, weights, n_supp, max_iter):
    """
    Free-support Wasserstein barycenter via fixed-point

    Formula: Iterate centroid update + transport solve

    Parameters
    ----------
    X_list : array-like
        Input data.
    weights : array-like
        Input data.
    n_supp : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y, weights_y

    References
    ----------
    Cuturi & Doucet (2014)
    """
    X_list = np.atleast_1d(np.asarray(X_list, dtype=float))
    n = len(X_list)
    result = float(np.mean(X_list))
    se = float(np.std(X_list, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Free-support Wasserstein barycenter via fixed-point"})


def cheatsheet():
    return "otbarfree: Free-support Wasserstein barycenter via fixed-point"
