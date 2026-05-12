# morie.fn -- function file (hadesllm/morie)
"""Strong approximation of DP posterior: coupling with Brownian bridge."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_strong_apx_dp"]


def ghosal_strong_apx_dp(x):
    """
    Strong approximation of DP posterior: coupling with Brownian bridge

    Formula: sup_t |sqrt(n)(G_n(t) - F_0(t)) - B(F_0(t))| -> 0 a.s.

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
    Ghosal Ch 12 §12.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Strong approximation of DP posterior: coupling with Brownian bridge"})


def cheatsheet():
    return "gh_c12_3: Strong approximation of DP posterior: coupling with Brownian bridge"
