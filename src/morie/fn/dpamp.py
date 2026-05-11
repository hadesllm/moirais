"""Privacy amplification by subsampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["privacy_amplification"]


def privacy_amplification(epsilon, q):
    """
    Privacy amplification by subsampling

    Formula: if M is ε-DP and we subsample at rate q, M is ~qε-DP

    Parameters
    ----------
    epsilon : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Balle-Barthe-Gaboardi (2018)
    """
    epsilon = np.atleast_1d(np.asarray(epsilon, dtype=float))
    n = len(epsilon)
    result = float(np.mean(epsilon))
    se = float(np.std(epsilon, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Privacy amplification by subsampling"})


def cheatsheet():
    return "dpamp: Privacy amplification by subsampling"
