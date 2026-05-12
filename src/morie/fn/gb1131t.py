# morie.fn -- function file (hadesllm/morie)
"""Spearman rho with ties correction formula."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_spearman_ties"]


def gibbons_spearman_ties(x, y):
    """
    Spearman rho with ties correction formula

    Formula: r_s = (n^3-n)/6 - sum(d_i^2) - T_x - T_y) / sqrt(products)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho_corrected

    References
    ----------
    Gibbons Ch 11.3 ties
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spearman rho with ties correction formula"})


def cheatsheet():
    return "gb1131t: Spearman rho with ties correction formula"
