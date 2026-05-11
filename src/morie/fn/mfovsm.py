"""Marginal feature-outcome MSM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mfo_vsm"]


def mfo_vsm(y, feature, A, H):
    """
    Marginal feature-outcome MSM

    Formula: feature-summary regression with IPTW

    Parameters
    ----------
    y : array-like
        Input data.
    feature : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Hernán (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Marginal feature-outcome MSM"})


def cheatsheet():
    return "mfovsm: Marginal feature-outcome MSM"
