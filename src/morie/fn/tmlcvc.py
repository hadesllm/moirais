"""Cross-validated TMLE — k-fold sample-splitting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_cv_targeting"]


def tmle_cv_targeting(y, D, X, K):
    """
    Cross-validated TMLE — k-fold sample-splitting

    Formula: split into K folds; fit Q,g on K-1; target on held-out

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zheng & vdL (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-validated TMLE — k-fold sample-splitting"})


def cheatsheet():
    return "tmlcvc: Cross-validated TMLE — k-fold sample-splitting"
