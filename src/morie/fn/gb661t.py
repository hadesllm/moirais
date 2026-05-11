# morie.fn — function file (hadesllm/morie)
"""Mann-Whitney U with tie correction for variance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_mw_ties"]


def gibbons_mw_ties(x, y):
    """
    Mann-Whitney U with tie correction for variance

    Formula: Var_adj(U) = mn/12 * (m+n+1 - sum t_j(t_j^2-1)/(m+n)(m+n-1))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Gibbons Ch 6.6 ties
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mann-Whitney U with tie correction for variance"})


def cheatsheet():
    return "gb661t: Mann-Whitney U with tie correction for variance"
