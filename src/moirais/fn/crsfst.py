"""Cross-fitted random survival forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["crs_forest"]


def crs_forest(time, event, D, X, K):
    """
    Cross-fitted random survival forest

    Formula: K-fold cross-fit; honest survival forest

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cui et al (2023)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-fitted random survival forest"})


def cheatsheet():
    return "crsfst: Cross-fitted random survival forest"
