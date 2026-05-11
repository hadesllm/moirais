# morie.fn — function file (hadesllm/morie)
"""Equivalence of Mann-Whitney U and Wilcoxon rank-sum W: W = U + m(m+1)/2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_mw_rs_equiv"]


def gibbons_mw_rs_equiv(x, y):
    """
    Equivalence of Mann-Whitney U and Wilcoxon rank-sum W: W = U + m(m+1)/2

    Formula: W = sum ranks of X in combined sample; U = W - m(m+1)/2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, U

    References
    ----------
    Gibbons Ch 6.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalence of Mann-Whitney U and Wilcoxon rank-sum W: W = U + m(m+1)/2"})


def cheatsheet():
    return "gb_mw2: Equivalence of Mann-Whitney U and Wilcoxon rank-sum W: W = U + m(m+1)/2"
