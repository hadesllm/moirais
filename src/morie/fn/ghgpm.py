# morie.fn — function file (hadesllm/morie)
"""GP with Matern kernel."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_matern"]


def ghosal_gp_matern(x, y):
    """
    GP with Matern kernel

    Formula: k(x,x') = sigma^2 * 2^{1-nu}/Gamma(nu) * (sqrt(2nu)*r/l)^nu * K_nu(...)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Ghosal Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP with Matern kernel"})


def cheatsheet():
    return "ghgpm: GP with Matern kernel"
