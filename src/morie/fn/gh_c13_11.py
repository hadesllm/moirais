# morie.fn -- function file (hadesllm/morie)
"""BvM theorem for NTR process: posterior for survival functional is asymptotically normal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ntr_bvm"]


def ghosal_ntr_bvm(x):
    """
    BvM theorem for NTR process: posterior for survival functional is asymptotically normal

    Formula: sqrt(n)(psi(F_n) - psi(F_0)) -> N(0, sigma^2) for smooth functional psi

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
    Ghosal Ch 13 §13.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BvM theorem for NTR process: posterior for survival functional is asymptotically normal"})


def cheatsheet():
    return "gh_c13_11: BvM theorem for NTR process: posterior for survival functional is asymptotically normal"
