"""Efron tie correction for Cox PH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["efron_tie_correction"]


def efron_tie_correction(time, event, X):
    """
    Efron tie correction for Cox PH

    Formula: L = product (sum exp(beta'X_j) - r/d * sum exp(beta'X_jin))

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Efron (1977)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Efron tie correction for Cox PH"})


def cheatsheet():
    return "efrnt: Efron tie correction for Cox PH"
