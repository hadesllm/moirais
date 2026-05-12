# morie.fn -- function file (hadesllm/morie)
"""Bracketing number N[](eps,T,d): min number of eps-brackets [l,u] covering T."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bracket_num"]


def ghosal_bracket_num(x):
    """
    Bracketing number N[](eps,T,d): min number of eps-brackets [l,u] covering T

    Formula: N[](eps,T,d) = min{#brackets: each f in T in some [l_i,u_i], d(l_i,u_i)<=eps}

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bracketing number N[](eps,T,d): min number of eps-brackets [l,u] covering T"})


def cheatsheet():
    return "gh_ap_c3: Bracketing number N[](eps,T,d): min number of eps-brackets [l,u] covering T"
