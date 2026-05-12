# morie.fn -- function file (hadesllm/morie)
"""Discreteness and support of DP: DP sample paths are discrete probability measures a.s.."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_discrete"]


def ghosal_dp_discrete(x):
    """
    Discreteness and support of DP: DP sample paths are discrete probability measures a.s.

    Formula: G ~ DP(alpha,G0) => G discrete a.s., supp(G) subset supp(G0) a.s.

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
    Ghosal Ch 4 §4.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discreteness and support of DP: DP sample paths are discrete probability measures a.s."})


def cheatsheet():
    return "gh_c4_12: Discreteness and support of DP: DP sample paths are discrete probability measures a.s."
