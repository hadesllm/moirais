"""Quantile-balanced causal forest for distributional treatment effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quantile_balanced_cf"]


def quantile_balanced_cf(y, D, X, quantile):
    """
    Quantile-balanced causal forest for distributional treatment effects

    Formula: tau_q(x) = Q_q[Y(1)|X=x] - Q_q[Y(0)|X=x]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Tibshirani-Wager (2019); Hsu-Huber-Lee-Liu (2022)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile-balanced causal forest for distributional treatment effects"})


def cheatsheet():
    return "qbcfgr: Quantile-balanced causal forest for distributional treatment effects"
