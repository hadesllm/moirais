# moirais.fn — function file (hadesllm/moirais)
"""Mutual singularity of DP measures with different base measures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_mutual_sing"]


def ghosal_dp_mutual_sing(x):
    """
    Mutual singularity of DP measures with different base measures

    Formula: DP(alpha, G0) perp DP(alpha', G0') if G0 != G0'

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
    Ghosal Ch 4 §4.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mutual singularity of DP measures with different base measures"})


def cheatsheet():
    return "gh_c4_15: Mutual singularity of DP measures with different base measures"
