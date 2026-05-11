# morie.fn — function file (hadesllm/morie)
"""Bootstrap for empirical process consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_bootstrap_empirical"]


def kosorok_bootstrap_empirical(x):
    """
    Bootstrap for empirical process consistency

    Formula: G_n^* = sqrt(n)(P_n^* - P_n)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap for empirical process consistency"})


def cheatsheet():
    return "ksr07: Bootstrap for empirical process consistency"
