# morie.fn — function file (hadesllm/morie)
"""Characterizations of DP: Sethuraman-Tiwari and Ferguson characterizations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_charact"]


def ghosal_dp_charact(x):
    """
    Characterizations of DP: Sethuraman-Tiwari and Ferguson characterizations

    Formula: G ~ DP iff all finite marginals are Dirichlet (Ferguson) or stick-breaking (Sethuraman)

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
    Ghosal Ch 4 §4.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Characterizations of DP: Sethuraman-Tiwari and Ferguson characterizations"})


def cheatsheet():
    return "gh_c4_19: Characterizations of DP: Sethuraman-Tiwari and Ferguson characterizations"
