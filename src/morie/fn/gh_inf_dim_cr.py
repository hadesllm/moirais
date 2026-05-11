# morie.fn — function file (hadesllm/morie)
"""Infinite-dimensional credible region: coverage under BvM conditions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_inf_dim_credible"]


def ghosal_inf_dim_credible(x):
    """
    Infinite-dimensional credible region: coverage under BvM conditions

    Formula: Pi_n(||theta - theta0||_H <= r_n | X^n) -> 1-alpha in P0^n probability

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
    Ghosal Ch 12 §12.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Infinite-dimensional credible region: coverage under BvM conditions"})


def cheatsheet():
    return "gh_inf_dim_cr: Infinite-dimensional credible region: coverage under BvM conditions"
