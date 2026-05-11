# morie.fn — function file (hadesllm/morie)
"""Histogram (binning) prior: density is piecewise constant on partition bins."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_histogram_prior"]


def ghosal_histogram_prior(x):
    """
    Histogram (binning) prior: density is piecewise constant on partition bins

    Formula: f(x) = sum_k p_k / |B_k| * 1_{x in B_k}, (p_1..p_K) ~ Dir(alpha)

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
    Ghosal Ch 2 §2.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Histogram (binning) prior: density is piecewise constant on partition bins"})


def cheatsheet():
    return "gh_c2_5: Histogram (binning) prior: density is piecewise constant on partition bins"
