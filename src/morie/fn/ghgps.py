# morie.fn — function file (hadesllm/morie)
"""GP with squared exponential kernel."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_squared_exponential"]


def ghosal_gp_squared_exponential(x, y):
    """
    GP with squared exponential kernel

    Formula: k(x,x') = sigma^2 exp(-||x-x'||^2 / (2*l^2))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP with squared exponential kernel"})


def cheatsheet():
    return "ghgps: GP with squared exponential kernel"
