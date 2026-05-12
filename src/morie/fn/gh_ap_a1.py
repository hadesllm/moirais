# morie.fn -- function file (hadesllm/morie)
"""Weak convergence of probability measures: integral f dP_n -> integral f dP for all f in Cb."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_weak_conv_def"]


def ghosal_weak_conv_def(x):
    """
    Weak convergence of probability measures: integral f dP_n -> integral f dP for all f in Cb

    Formula: P_n ->_w P iff E_{P_n}[f] -> E_P[f] for all bounded continuous f

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
    Ghosal App A
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weak convergence of probability measures: integral f dP_n -> integral f dP for all f in Cb"})


def cheatsheet():
    return "gh_ap_a1: Weak convergence of probability measures: integral f dP_n -> integral f dP for all f in Cb"
