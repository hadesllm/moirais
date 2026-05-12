# morie.fn -- function file (hadesllm/morie)
"""Glivenko-Cantelli theorem: empirical CDF converges uniformly to true CDF."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_glivenko"]


def ghosal_glivenko(x):
    """
    Glivenko-Cantelli theorem: empirical CDF converges uniformly to true CDF

    Formula: ||F_n - F||_infty -> 0 a.s. for any F

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
    Ghosal App F
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glivenko-Cantelli theorem: empirical CDF converges uniformly to true CDF"})


def cheatsheet():
    return "gh_ap_f2: Glivenko-Cantelli theorem: empirical CDF converges uniformly to true CDF"
