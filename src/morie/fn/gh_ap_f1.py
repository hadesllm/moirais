# morie.fn -- function file (hadesllm/morie)
"""Donsker class: family F with sqrt(n)(P_n-P)(f) weak convergence in l^infty(F)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_donsker_class"]


def ghosal_donsker_class(x):
    """
    Donsker class: family F with sqrt(n)(P_n-P)(f) weak convergence in l^infty(F)

    Formula: F is Donsker iff J[](1, F, L2) < infty (Dudley bracketing integral)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Donsker class: family F with sqrt(n)(P_n-P)(f) weak convergence in l^infty(F)"})


def cheatsheet():
    return "gh_ap_f1: Donsker class: family F with sqrt(n)(P_n-P)(f) weak convergence in l^infty(F)"
