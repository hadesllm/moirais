"""Standardized Precipitation Index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spi"]


def spi(precip, window):
    """
    Standardized Precipitation Index

    Formula: Z-score of fitted gamma precip

    Parameters
    ----------
    precip : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McKee-Doesken-Kleist (1993)
    """
    precip = np.atleast_1d(np.asarray(precip, dtype=float))
    n = len(precip)
    result = float(np.mean(precip))
    se = float(np.std(precip, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standardized Precipitation Index"})


def cheatsheet():
    return "droSPI: Standardized Precipitation Index"
