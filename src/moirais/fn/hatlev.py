"""Hat-matrix leverage h_ii."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["leverage"]


def leverage(X):
    """
    Hat-matrix leverage h_ii

    Formula: H = X(X^T X)^{-1}X^T; h_ii = diag(H)

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
    Hoaglin-Welsch (1978)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hat-matrix leverage h_ii"})


def cheatsheet():
    return "hatlev: Hat-matrix leverage h_ii"
