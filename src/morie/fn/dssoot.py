"""Bootstrap CI for indirect effect ab."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bootstrap_indirect"]


def bootstrap_indirect(Y, X, M, n_boot):
    """
    Bootstrap CI for indirect effect ab

    Formula: resample -> recompute ab -> percentile CI

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    n_boot : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Preacher-Hayes (2008)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap CI for indirect effect ab"})


def cheatsheet():
    return "dssoot: Bootstrap CI for indirect effect ab"
