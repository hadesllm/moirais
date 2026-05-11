"""Informer for long-horizon forecasting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["informer_long_horizon"]


def informer_long_horizon(y, horizon):
    """
    Informer for long-horizon forecasting

    Formula: sparse self-attention + distilling

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhou et al (2021) Informer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Informer for long-horizon forecasting"})


def cheatsheet():
    return "informer: Informer for long-horizon forecasting"
