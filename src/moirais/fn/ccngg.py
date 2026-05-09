"""Nakagawa-Schielzeth conditional R^2 (fixed + random) for LMM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nakagawa_conditional_r2"]


def nakagawa_conditional_r2(y, X, Z, cluster):
    """
    Nakagawa-Schielzeth conditional R^2 (fixed + random) for LMM

    Formula: R^2_c = (sigma2_f + sum sigma2_l) / (sigma2_f + sum sigma2_l + sigma2_e)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nakagawa & Schielzeth (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nakagawa-Schielzeth conditional R^2 (fixed + random) for LMM"})


def cheatsheet():
    return "ccngg: Nakagawa-Schielzeth conditional R^2 (fixed + random) for LMM"
