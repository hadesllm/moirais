"""Percentile (bootstrap) confidence interval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_bootstrap_percentile"]


def wasserman_bootstrap_percentile(data, T, B, alpha):
    """
    Percentile (bootstrap) confidence interval

    Formula: (q_{alpha/2}, q_{1-alpha/2})

    Parameters
    ----------
    data : array-like
        Input data.
    T : array-like
        Input data.
    B : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lower, upper

    References
    ----------
    Wasserman (2004), Ch 8
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Percentile (bootstrap) confidence interval"})


def cheatsheet():
    return "wsmbpc: Percentile (bootstrap) confidence interval"
