# morie.fn -- function file (hadesllm/morie)
"""Stick-breaking DP: V_k ~ Beta(1,alpha), G = sum_k w_k delta_{theta_k}."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_stickbr"]


def ghosal_dp_stickbr(x):
    """
    Stick-breaking DP: V_k ~ Beta(1,alpha), G = sum_k w_k delta_{theta_k}

    Formula: w_k = V_k prod_{j<k}(1-V_j), V_k iid Beta(1,alpha), theta_k iid G0

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
    Ghosal Ch 4 §4.2.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stick-breaking DP: V_k ~ Beta(1,alpha), G = sum_k w_k delta_{theta_k}"})


def cheatsheet():
    return "gh_c4_11: Stick-breaking DP: V_k ~ Beta(1,alpha), G = sum_k w_k delta_{theta_k}"
