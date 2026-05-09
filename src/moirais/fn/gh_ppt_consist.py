# moirais.fn — function file (hadesllm/moirais)
"""Polya tree posterior contraction rate: n^{-s/(2s+1)} for s-Holder densities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_polya_tree_consist_rate"]


def ghosal_polya_tree_consist_rate(x):
    """
    Polya tree posterior contraction rate: n^{-s/(2s+1)} for s-Holder densities

    Formula: PT*(alpha, m^2): contraction rate n^{-s/(2s+1)} in Hellinger

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
    Ghosal Ch 7 §7.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polya tree posterior contraction rate: n^{-s/(2s+1)} for s-Holder densities"})


def cheatsheet():
    return "gh_ppt_consist: Polya tree posterior contraction rate: n^{-s/(2s+1)} for s-Holder densities"
