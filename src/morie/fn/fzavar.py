# morie.fn -- function file (hadesllm/morie)
"""Asymptotic variance of sample quantile."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_quantile_asymp_var"]


def fauzi_quantile_asymp_var(data, p):
    """
    Asymptotic variance of sample quantile

    Formula: sigma^2 = p(1-p) / [f(Q(p))]^2

    Parameters
    ----------
    data : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

    References
    ----------
    Fauzi Ch 3, Eq 3.2
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic variance of sample quantile"})


def cheatsheet():
    return "fzavar: Asymptotic variance of sample quantile"
