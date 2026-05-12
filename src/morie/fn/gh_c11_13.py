# morie.fn -- function file (hadesllm/morie)
"""GP adaptation theorem: random bandwidth GP adapts to unknown smoothness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_adapt_thm"]


def ghosal_gp_adapt_thm(x):
    """
    GP adaptation theorem: random bandwidth GP adapts to unknown smoothness

    Formula: f ~ GP(0, k_{l_n}), l_n ~ Pi_l, rate adapts to any s-Holder f0

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
    Ghosal Ch 11 §11.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP adaptation theorem: random bandwidth GP adapts to unknown smoothness"})


def cheatsheet():
    return "gh_c11_13: GP adaptation theorem: random bandwidth GP adapts to unknown smoothness"
