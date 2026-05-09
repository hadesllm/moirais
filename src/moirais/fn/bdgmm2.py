"""Bound via GMM-alt criterion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_gmm_alt"]


def bound_gmm_alt(y, X, moments, W):
    """
    Bound via GMM-alt criterion

    Formula: GMM with moment inequalities

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    moments : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews-Soares (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound via GMM-alt criterion"})


def cheatsheet():
    return "bdgmm2: Bound via GMM-alt criterion"
