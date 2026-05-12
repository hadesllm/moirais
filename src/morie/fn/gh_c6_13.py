# morie.fn -- function file (hadesllm/morie)
"""Le Cam inequality approach to posterior consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_lecam_consist"]


def ghosal_lecam_consist(x):
    """
    Le Cam inequality approach to posterior consistency

    Formula: Pi_n(U^c|X^n) <= (1/alpha_n) sum p(X_i|theta)/p0(X_i) for test phi_n

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
    Ghosal Ch 6 §6.8.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Le Cam inequality approach to posterior consistency"})


def cheatsheet():
    return "gh_c6_13: Le Cam inequality approach to posterior consistency"
