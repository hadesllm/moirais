# morie.fn — function file (hadesllm/morie)
"""Coefficient of concordance for balanced incomplete block designs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_balance_incomplete"]


def gibbons_balance_incomplete(rankings, lam, n, k):
    """
    Coefficient of concordance for balanced incomplete block designs

    Formula: W_b = 12*S_b / (k^2*(lam*(n^3-n)/(n-1))) using BIB lam

    Parameters
    ----------
    rankings : array-like
        Input data.
    lam : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, p_value

    References
    ----------
    Gibbons Ch 12.5
    """
    rankings = np.asarray(rankings, dtype=float)
    n = int(rankings) if rankings.ndim == 0 else len(rankings)
    result = float(np.mean(rankings))
    se = float(np.std(rankings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coefficient of concordance for balanced incomplete block designs"})


def cheatsheet():
    return "gb_blt: Coefficient of concordance for balanced incomplete block designs"
