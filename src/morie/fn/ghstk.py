# morie.fn — function file (hadesllm/morie)
"""Truncated stick-breaking representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_stick_breaking_trunc"]


def ghosal_stick_breaking_trunc(x):
    """
    Truncated stick-breaking representation

    Formula: G_K = sum_{k=1}^K w_k delta_{theta_k}

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
    Ghosal Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Truncated stick-breaking representation"})


def cheatsheet():
    return "ghstk: Truncated stick-breaking representation"
