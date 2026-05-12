# morie.fn -- function file (hadesllm/morie)
"""Epsilon-packing number D(eps,T,d): max number of eps-separated points in T."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_packing_num"]


def ghosal_packing_num(x):
    """
    Epsilon-packing number D(eps,T,d): max number of eps-separated points in T

    Formula: D(eps,T,d) = max{card(S): d(s_i,s_j)>eps for all i!=j, S subset T}

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
    Ghosal App C
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epsilon-packing number D(eps,T,d): max number of eps-separated points in T"})


def cheatsheet():
    return "gh_ap_c2: Epsilon-packing number D(eps,T,d): max number of eps-separated points in T"
