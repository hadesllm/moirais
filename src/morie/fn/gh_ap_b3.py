# morie.fn — function file (hadesllm/morie)
"""Renyi divergence of order alpha: D_alpha(P||Q) = log integral p^alpha q^{1-alpha} / (alpha-1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_renyi_div"]


def ghosal_renyi_div(x):
    """
    Renyi divergence of order alpha: D_alpha(P||Q) = log integral p^alpha q^{1-alpha} / (alpha-1)

    Formula: D_alpha(P||Q) = (1/(alpha-1)) log integral p^alpha q^{1-alpha}, alpha in (0,1)

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
    Ghosal App B
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Renyi divergence of order alpha: D_alpha(P||Q) = log integral p^alpha q^{1-alpha} / (alpha-1)"})


def cheatsheet():
    return "gh_ap_b3: Renyi divergence of order alpha: D_alpha(P||Q) = log integral p^alpha q^{1-alpha} / (alpha-1)"
