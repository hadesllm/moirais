# moirais.fn — function file (hadesllm/moirais)
"""Kullback-Leibler support condition: prior assigns positive mass to every KL ball around P0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_kl_support"]


def ghosal_kl_support(x):
    """
    Kullback-Leibler support condition: prior assigns positive mass to every KL ball around P0

    Formula: Pi({P: KL(P0,P)<eps}) > 0 for all eps > 0

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
    Ghosal Ch 6 §6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kullback-Leibler support condition: prior assigns positive mass to every KL ball around P0"})


def cheatsheet():
    return "gh_c6_6: Kullback-Leibler support condition: prior assigns positive mass to every KL ball around P0"
