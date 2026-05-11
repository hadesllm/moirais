# morie.fn — function file (hadesllm/morie)
"""Variance of DP: Var[G(A)] = G0(A)(1-G0(A))/(alpha+1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_var"]


def ghosal_dp_var(x):
    """
    Variance of DP: Var[G(A)] = G0(A)(1-G0(A))/(alpha+1)

    Formula: Var[G(A)] = G0(A)(1-G0(A)) / (alpha+1)

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
    Ghosal Ch 4 §4.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of DP: Var[G(A)] = G0(A)(1-G0(A))/(alpha+1)"})


def cheatsheet():
    return "gh_c4_3: Variance of DP: Var[G(A)] = G0(A)(1-G0(A))/(alpha+1)"
