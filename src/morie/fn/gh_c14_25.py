# morie.fn -- function file (hadesllm/morie)
"""Poisson process representation of IBP: features as points of Poisson process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ibp_poisson"]


def ghosal_ibp_poisson(x):
    """
    Poisson process representation of IBP: features as points of Poisson process

    Formula: Features = Poisson process on [0,1] with intensity alpha, thinned per customer

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
    Ghosal Ch 14 §14.10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson process representation of IBP: features as points of Poisson process"})


def cheatsheet():
    return "gh_c14_25: Poisson process representation of IBP: features as points of Poisson process"
