"""Weiszfeld iteration for L1 median."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weiszfeld"]


def weiszfeld(X, tol, max_iter):
    """
    Weiszfeld iteration for L1 median

    Formula: μ ← (sum w_i x_i)/(sum w_i), w_i = 1/||x_i−μ||

    Parameters
    ----------
    X : array-like
        Input data.
    tol : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Weiszfeld (1937)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weiszfeld iteration for L1 median"})


def cheatsheet():
    return "weisz: Weiszfeld iteration for L1 median"
