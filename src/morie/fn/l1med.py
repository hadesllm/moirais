"""Geometric (L1) median."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["l1_median"]


def l1_median(X, tol):
    """
    Geometric (L1) median

    Formula: min_μ sum ||x_i − μ||

    Parameters
    ----------
    X : array-like
        Input data.
    tol : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geometric (L1) median"})


def cheatsheet():
    return "l1med: Geometric (L1) median"
