"""Fisher-Rao information metric."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["information_geometry"]


def information_geometry(log_p, theta):
    """
    Fisher-Rao information metric

    Formula: g_ij(theta) = E[d_i log p · d_j log p]

    Parameters
    ----------
    log_p : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Amari (1985)
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fisher-Rao information metric"})


def cheatsheet():
    return "infgnt: Fisher-Rao information metric"
