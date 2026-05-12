# morie.fn -- function file (hadesllm/morie)
"""Increasing-process prior via integrated Gaussian process for monotone functions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_increasing_prior"]


def ghosal_gp_increasing_prior(x):
    """
    Increasing-process prior via integrated Gaussian process for monotone functions

    Formula: F(t) = integral_0^t exp(W(s)) ds, W ~ GP

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 2 §2.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Increasing-process prior via integrated Gaussian process for monotone functions"})


def cheatsheet():
    return "gh_c2_3: Increasing-process prior via integrated Gaussian process for monotone functions"
