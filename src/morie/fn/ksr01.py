# morie.fn — function file (hadesllm/morie)
"""Empirical process indexed by function class."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_empirical_process"]


def kosorok_empirical_process(x, f):
    """
    Empirical process indexed by function class

    Formula: G_n(f) = sqrt(n)(P_n - P)(f)

    Parameters
    ----------
    x : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical process indexed by function class"})


def cheatsheet():
    return "ksr01: Empirical process indexed by function class"
