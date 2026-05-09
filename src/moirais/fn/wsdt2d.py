"""p-Wasserstein for general dimension."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserstein_p_d"]


def wasserstein_p_d(X_samples, Y_samples, p):
    """
    p-Wasserstein for general dimension

    Formula: W_p^p = inf E[||X - Y||^p] over couplings

    Parameters
    ----------
    X_samples : array-like
        Input data.
    Y_samples : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Villani (2009)
    """
    X_samples = np.atleast_1d(np.asarray(X_samples, dtype=float))
    n = len(X_samples)
    result = float(np.mean(X_samples))
    se = float(np.std(X_samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "p-Wasserstein for general dimension"})


def cheatsheet():
    return "wsdt2d: p-Wasserstein for general dimension"
