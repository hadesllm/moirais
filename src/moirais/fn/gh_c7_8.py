# moirais.fn — function file (hadesllm/moirais)
"""Semiparametric consistency for location problem: location + unknown error density."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_loc_semipara"]


def ghosal_loc_semipara(x):
    """
    Semiparametric consistency for location problem: location + unknown error density

    Formula: Y_i = theta + e_i, e_i ~ f, both theta and f estimated

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
    Ghosal Ch 7 §7.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric consistency for location problem: location + unknown error density"})


def cheatsheet():
    return "gh_c7_8: Semiparametric consistency for location problem: location + unknown error density"
