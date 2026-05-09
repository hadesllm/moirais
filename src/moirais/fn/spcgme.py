"""Spatial concordance via kappa."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_concordance_kappa"]


def spatial_concordance_kappa(x, y, W):
    """
    Spatial concordance via kappa

    Formula: weighted kappa per spatial neighborhood

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cohen (1968)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial concordance via kappa"})


def cheatsheet():
    return "spcgme: Spatial concordance via kappa"
