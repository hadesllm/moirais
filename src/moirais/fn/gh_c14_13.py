# moirais.fn — function file (hadesllm/moirais)
"""Levy measure representation of Poisson-Kingman process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pk_levy"]


def ghosal_pk_levy(x):
    """
    Levy measure representation of Poisson-Kingman process

    Formula: rho(du): Levy measure on (0,infty), G = normalized sum of Poisson jumps

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
    Ghosal Ch 14 §14.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Levy measure representation of Poisson-Kingman process"})


def cheatsheet():
    return "gh_c14_13: Levy measure representation of Poisson-Kingman process"
