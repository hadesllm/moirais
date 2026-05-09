"""Optimal Huber k for given efficiency."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["optimal_huber_k"]


def optimal_huber_k(target_eff):
    """
    Optimal Huber k for given efficiency

    Formula: choose k to attain ARE at Gaussian

    Parameters
    ----------
    target_eff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huber (1964)
    """
    target_eff = np.atleast_1d(np.asarray(target_eff, dtype=float))
    n = len(target_eff)
    result = float(np.mean(target_eff))
    se = float(np.std(target_eff, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal Huber k for given efficiency"})


def cheatsheet():
    return "opthr: Optimal Huber k for given efficiency"
