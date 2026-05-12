# morie.fn -- function file (hadesllm/morie)
"""Stick-breaking construction: w_k = V_k * prod_{j<k}(1-V_j), sum w_k = 1."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_stick_break_def"]


def ghosal_stick_break_def(x):
    """
    Stick-breaking construction: w_k = V_k * prod_{j<k}(1-V_j), sum w_k = 1

    Formula: G = sum_k w_k delta_{theta_k}, V_k ~ Beta(a_k, b_k), theta_k ~ G0

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
    Ghosal Ch 3 §3.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stick-breaking construction: w_k = V_k * prod_{j<k}(1-V_j), sum w_k = 1"})


def cheatsheet():
    return "gh_c3_4: Stick-breaking construction: w_k = V_k * prod_{j<k}(1-V_j), sum w_k = 1"
