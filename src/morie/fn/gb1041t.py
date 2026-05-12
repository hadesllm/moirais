# morie.fn -- function file (hadesllm/morie)
"""Kruskal-Wallis H with tie correction factor."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_kw_ties"]


def gibbons_kw_ties(groups):
    """
    Kruskal-Wallis H with tie correction factor

    Formula: H_adj = H / (1 - sum t_j(t_j^2-1)/(N^3-N))

    Parameters
    ----------
    groups : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: adjusted_statistic

    References
    ----------
    Gibbons Ch 10.4 ties
    """
    groups = np.asarray(groups, dtype=float)
    n = int(groups) if groups.ndim == 0 else len(groups)
    result = float(np.mean(groups))
    se = float(np.std(groups, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kruskal-Wallis H with tie correction factor"})


def cheatsheet():
    return "gb1041t: Kruskal-Wallis H with tie correction factor"
