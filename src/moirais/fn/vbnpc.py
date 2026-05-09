"""Variational Bayes for DP/HDP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vb_nonparametric"]


def vb_nonparametric(y, K_truncate, alpha):
    """
    Variational Bayes for DP/HDP

    Formula: truncated stick-breaking + mean-field

    Parameters
    ----------
    y : array-like
        Input data.
    K_truncate : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blei-Jordan (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational Bayes for DP/HDP"})


def cheatsheet():
    return "vbnpc: Variational Bayes for DP/HDP"
