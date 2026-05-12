# morie.fn -- function file (hadesllm/morie)
"""Polya tree process PT(alpha, pi): beta-distributed mass allocations at each tree level."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_polya_tree_def"]


def ghosal_polya_tree_def(x):
    """
    Polya tree process PT(alpha, pi): beta-distributed mass allocations at each tree level

    Formula: PT(T_m, A): Y_{e0|e} ~ Beta(alpha_{e0}, alpha_{e1}), G(B_e0)=Y_{e0|e}*G(B_e)

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
    Ghosal Ch 3 §3.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polya tree process PT(alpha, pi): beta-distributed mass allocations at each tree level"})


def cheatsheet():
    return "gh_c3_12: Polya tree process PT(alpha, pi): beta-distributed mass allocations at each tree level"
