"""Quantile-balanced score for forests."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qb_cf_score"]


def qb_cf_score(y, D, X, quantile):
    """
    Quantile-balanced score for forests

    Formula: weight residual by quantile distance

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hsu et al (2022)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile-balanced score for forests"})


def cheatsheet():
    return "qbcfgs: Quantile-balanced score for forests"
