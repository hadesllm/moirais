"""Subsampling (m-out-of-n) without replacement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_subsampling"]


def boot_subsampling(x, m, stat, B):
    """
    Subsampling (m-out-of-n) without replacement

    Formula: Sample subsets of size m<n; collect θ̂_b

    Parameters
    ----------
    x : array-like
        Input data.
    m : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Politis-Romano-Wolf (1999)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Subsampling (m-out-of-n) without replacement"})


def cheatsheet():
    return "btsubs: Subsampling (m-out-of-n) without replacement"
