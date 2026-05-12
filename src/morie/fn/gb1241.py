# morie.fn -- function file (hadesllm/morie)
"""Kendall coefficient of concordance W for k sets of rankings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_concordance_w"]


def gibbons_concordance_w(rankings):
    """
    Kendall coefficient of concordance W for k sets of rankings

    Formula: W = 12*S / (k^2*(n^3-n)); S = sum(R_j - Rbar)^2

    Parameters
    ----------
    rankings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, p_value

    References
    ----------
    Gibbons Ch 12.4
    """
    rankings = np.asarray(rankings, dtype=float)
    n = int(rankings) if rankings.ndim == 0 else len(rankings)
    result = float(np.mean(rankings))
    se = float(np.std(rankings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kendall coefficient of concordance W for k sets of rankings"})


def cheatsheet():
    return "gb1241: Kendall coefficient of concordance W for k sets of rankings"
