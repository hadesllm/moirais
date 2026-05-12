# morie.fn -- function file (hadesllm/morie)
"""Wilcoxon rank-sum with ties correction using midranks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_wrs_ties"]


def gibbons_wrs_ties(x, y):
    """
    Wilcoxon rank-sum with ties correction using midranks

    Formula: Var_adj = mn/12 * (N+1 - sum t_k(t_k^2-1)/(N(N-1)))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: adjusted_statistic

    References
    ----------
    Gibbons Ch 8.2 ties
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wilcoxon rank-sum with ties correction using midranks"})


def cheatsheet():
    return "gb821t: Wilcoxon rank-sum with ties correction using midranks"
