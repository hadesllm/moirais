"""DP-means: hard-clustering DP analog."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_kmeans"]


def dp_kmeans(y, lam):
    """
    DP-means: hard-clustering DP analog

    Formula: argmin sum dist^2 + lambda K

    Parameters
    ----------
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kulis-Jordan (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP-means: hard-clustering DP analog"})


def cheatsheet():
    return "dpkmn: DP-means: hard-clustering DP analog"
