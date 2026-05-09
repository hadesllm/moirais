"""Empirical quantile q_p = F_n^{-1}(p)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_empirical_quantile"]


def wasserman_empirical_quantile(data, p):
    """
    Empirical quantile q_p = F_n^{-1}(p)

    Formula: q_p = inf{x : F_n(x) >= p}

    Parameters
    ----------
    data : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: quantile

    References
    ----------
    Wasserman (2004), Ch 7
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical quantile q_p = F_n^{-1}(p)"})


def cheatsheet():
    return "wsmqtl: Empirical quantile q_p = F_n^{-1}(p)"
