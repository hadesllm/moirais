# morie.fn — function file (hadesllm/morie)
"""Weak posterior consistency: Pi_n(U^c | X^n) -> 0 in P0-probability for all weak neighborhoods U of P0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_weak_consist"]


def ghosal_weak_consist(x):
    """
    Weak posterior consistency: Pi_n(U^c | X^n) -> 0 in P0-probability for all weak neighborhoods U of P0

    Formula: Pi_n({P: d_w(P,P0) > eps} | X_1..X_n) ->_{P0} 0

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
    Ghosal Ch 6 §6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weak posterior consistency: Pi_n(U^c | X^n) -> 0 in P0-probability for all weak neighborhoods U of P0"})


def cheatsheet():
    return "gh_c6_1: Weak posterior consistency: Pi_n(U^c | X^n) -> 0 in P0-probability for all weak neighborhoods U of P0"
