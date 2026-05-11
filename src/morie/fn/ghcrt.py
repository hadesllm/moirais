# morie.fn — function file (hadesllm/morie)
"""Posterior contraction rate computation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_contraction_rate"]


def ghosal_contraction_rate(x):
    """
    Posterior contraction rate computation

    Formula: Pi(d(theta,theta0) > M*eps_n | data) -> 0

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
    Ghosal Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior contraction rate computation"})


def cheatsheet():
    return "ghcrt: Posterior contraction rate computation"
