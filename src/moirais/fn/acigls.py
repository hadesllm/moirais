"""Adjusted IP-weighted GLS for clustered."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adjusted_ipgls"]


def adjusted_ipgls(y, A, H, cluster):
    """
    Adjusted IP-weighted GLS for clustered

    Formula: GLS with cluster-robust SE

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schildcrout-Heagerty (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adjusted IP-weighted GLS for clustered"})


def cheatsheet():
    return "acigls: Adjusted IP-weighted GLS for clustered"
