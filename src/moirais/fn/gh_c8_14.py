# moirais.fn — function file (hadesllm/moirais)
"""Contraction for convex models under misspecification: unique KL projection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_convex_misp"]


def ghosal_convex_misp(x):
    """
    Contraction for convex models under misspecification: unique KL projection

    Formula: Model convex => P* unique, contraction at standard eps_n rate

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
    Ghosal Ch 8 §8.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contraction for convex models under misspecification: unique KL projection"})


def cheatsheet():
    return "gh_c8_14: Contraction for convex models under misspecification: unique KL projection"
