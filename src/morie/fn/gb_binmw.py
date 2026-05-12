# morie.fn -- function file (hadesllm/morie)
"""Mann-Whitney U relation to Wilcoxon rank-sum W: U = W - m(m+1)/2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_mw_binomial_link"]


def gibbons_mw_binomial_link(W, m):
    """
    Mann-Whitney U relation to Wilcoxon rank-sum W: U = W - m(m+1)/2

    Formula: U = W - m(m+1)/2; two statistics differ by a constant function of m

    Parameters
    ----------
    W : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: U

    References
    ----------
    Gibbons Ch 6.6
    """
    W = np.asarray(W, dtype=float)
    n = int(W) if W.ndim == 0 else len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mann-Whitney U relation to Wilcoxon rank-sum W: U = W - m(m+1)/2"})


def cheatsheet():
    return "gb_binmw: Mann-Whitney U relation to Wilcoxon rank-sum W: U = W - m(m+1)/2"
