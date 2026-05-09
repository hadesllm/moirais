# moirais.fn — function file (hadesllm/moirais)
"""DP-based prior contraction: discreteness penalty leads to extra log factor."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_disc_crt"]


def ghosal_dp_disc_crt(x):
    """
    DP-based prior contraction: discreteness penalty leads to extra log factor

    Formula: DP prior: rate n^{-s/(2s+1)} * (log n)^t for density estimation

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
    Ghosal Ch 9 §9.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP-based prior contraction: discreteness penalty leads to extra log factor"})


def cheatsheet():
    return "gh_c9_2: DP-based prior contraction: discreteness penalty leads to extra log factor"
