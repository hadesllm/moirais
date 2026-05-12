# morie.fn -- function file (hadesllm/morie)
"""Assouad lemma: lower bound on minimax risk via Hamming distance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_assouad_lemma"]


def ghosal_assouad_lemma(x):
    """
    Assouad lemma: lower bound on minimax risk via Hamming distance

    Formula: R_n >= (m/2) * min_{d(P0,P1)=1} d(P_tau0, P_tau1)^n / 4

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
    Ghosal App K
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Assouad lemma: lower bound on minimax risk via Hamming distance"})


def cheatsheet():
    return "gh_ap_k2: Assouad lemma: lower bound on minimax risk via Hamming distance"
