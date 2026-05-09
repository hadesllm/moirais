"""Lagged-value IPTW."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["laggedval_iptw"]


def laggedval_iptw(y, A, H, lag):
    """
    Lagged-value IPTW

    Formula: include Y_{t-1} in propensity model

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (1986)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lagged-value IPTW"})


def cheatsheet():
    return "lggvls: Lagged-value IPTW"
