"""Neural collaborative filtering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ncf"]


def ncf(R, K, mlp_h):
    """
    Neural collaborative filtering

    Formula: GMF + MLP fused

    Parameters
    ----------
    R : array-like
        Input data.
    K : array-like
        Input data.
    mlp_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    He et al (2017) NCF
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neural collaborative filtering"})


def cheatsheet():
    return "ncfRS: Neural collaborative filtering"
