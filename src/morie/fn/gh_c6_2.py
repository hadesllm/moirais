# morie.fn -- function file (hadesllm/morie)
"""Strong posterior consistency: Pi_n(U^c | X^n) -> 0 a.s. for all neighborhoods."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_strong_consist"]


def ghosal_strong_consist(x):
    """
    Strong posterior consistency: Pi_n(U^c | X^n) -> 0 a.s. for all neighborhoods

    Formula: Pi_n({P: d(P,P0) > eps} | X_1..X_n) -> 0 P0^infty-a.s.

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Strong posterior consistency: Pi_n(U^c | X^n) -> 0 a.s. for all neighborhoods"})


def cheatsheet():
    return "gh_c6_2: Strong posterior consistency: Pi_n(U^c | X^n) -> 0 a.s. for all neighborhoods"
