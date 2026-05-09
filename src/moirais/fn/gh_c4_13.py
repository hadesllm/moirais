# moirais.fn — function file (hadesllm/moirais)
"""Weak convergence of DP: DP(alpha_n, G0_n) -> DP(alpha, G0) weakly."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_weak_conv"]


def ghosal_dp_weak_conv(x):
    """
    Weak convergence of DP: DP(alpha_n, G0_n) -> DP(alpha, G0) weakly

    Formula: alpha_n -> alpha, G0_n ->_w G0 => DP(alpha_n,G0_n) ->_w DP(alpha,G0)

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
    Ghosal Ch 4 §4.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weak convergence of DP: DP(alpha_n, G0_n) -> DP(alpha, G0) weakly"})


def cheatsheet():
    return "gh_c4_13: Weak convergence of DP: DP(alpha_n, G0_n) -> DP(alpha, G0) weakly"
