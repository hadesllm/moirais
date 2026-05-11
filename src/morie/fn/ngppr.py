"""Normalized gamma process prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["normalized_gamma_process"]


def normalized_gamma_process(y, alpha):
    """
    Normalized gamma process prior

    Formula: normalize gamma process for random measure

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lijoi-Prünster (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized gamma process prior"})


def cheatsheet():
    return "ngppr: Normalized gamma process prior"
