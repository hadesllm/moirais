"""Oceanic Niño Index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["oni"]


def oni(sst_n34):
    """
    Oceanic Niño Index

    Formula: 3-month running mean Niño 3.4 anomaly

    Parameters
    ----------
    sst_n34 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    NOAA CPC
    """
    sst_n34 = np.atleast_1d(np.asarray(sst_n34, dtype=float))
    n = len(sst_n34)
    result = float(np.mean(sst_n34))
    se = float(np.std(sst_n34, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Oceanic Niño Index"})


def cheatsheet():
    return "oniIdx: Oceanic Niño Index"
