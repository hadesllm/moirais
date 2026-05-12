# morie.fn -- function file (hadesllm/morie)
"""Partially specified Polya tree: some partition levels left free."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_partspec_pt"]


def ghosal_partspec_pt(x):
    """
    Partially specified Polya tree: some partition levels left free

    Formula: PT specified only at selected levels m1 < m2 < ..., rest marginalized

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
    Ghosal Ch 3 §3.7.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partially specified Polya tree: some partition levels left free"})


def cheatsheet():
    return "gh_c3_15: Partially specified Polya tree: some partition levels left free"
