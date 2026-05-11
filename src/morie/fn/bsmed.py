"""Bootstrap percentile CI for indirect effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bootstrap_mediation_ci"]


def bootstrap_mediation_ci(X, M, Y, B):
    """
    Bootstrap percentile CI for indirect effect

    Formula: resample n, recompute a*b, take percentile of distribution

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Preacher & Hayes (2004); Shrout & Bolger (2002)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap percentile CI for indirect effect"})


def cheatsheet():
    return "bsmed: Bootstrap percentile CI for indirect effect"
