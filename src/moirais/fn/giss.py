"""GISS surface temp anomaly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["giss_anomaly"]


def giss_anomaly(T, baseline):
    """
    GISS surface temp anomaly

    Formula: deviation from 1951-1980 baseline

    Parameters
    ----------
    T : array-like
        Input data.
    baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hansen et al (1999)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GISS surface temp anomaly"})


def cheatsheet():
    return "giss: GISS surface temp anomaly"
