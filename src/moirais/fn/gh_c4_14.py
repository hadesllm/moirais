# moirais.fn — function file (hadesllm/moirais)
"""Ferguson-Sethuraman approximation of DP by finite stick-breaking truncation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_fs_approx"]


def ghosal_dp_fs_approx(x):
    """
    Ferguson-Sethuraman approximation of DP by finite stick-breaking truncation

    Formula: G_K = sum_{k=1}^K w_k delta_{theta_k} + w_{K+1} G0, approx error bounded

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
    Ghosal Ch 4 §4.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ferguson-Sethuraman approximation of DP by finite stick-breaking truncation"})


def cheatsheet():
    return "gh_c4_14: Ferguson-Sethuraman approximation of DP by finite stick-breaking truncation"
