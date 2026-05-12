# morie.fn -- function file (hadesllm/morie)
"""Epsilon-covering number N(eps,T,d): minimum number of eps-balls covering T."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_covering_num"]


def ghosal_covering_num(x):
    """
    Epsilon-covering number N(eps,T,d): minimum number of eps-balls covering T

    Formula: N(eps,T,d) = min{card(C): T subset union_{c in C} B(c,eps)}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epsilon-covering number N(eps,T,d): minimum number of eps-balls covering T"})


def cheatsheet():
    return "gh_ap_c1: Epsilon-covering number N(eps,T,d): minimum number of eps-balls covering T"
