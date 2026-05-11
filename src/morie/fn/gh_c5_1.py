# morie.fn — function file (hadesllm/morie)
"""DP mixture model: f(x) = integral phi(x|theta) dG(theta), G ~ DP(alpha, G0)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dpm_model"]


def ghosal_dpm_model(x):
    """
    DP mixture model: f(x) = integral phi(x|theta) dG(theta), G ~ DP(alpha, G0)

    Formula: f(x) = integral K(x;theta) dG(theta), G ~ DP(alpha,G0)

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
    Ghosal Ch 5 §5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP mixture model: f(x) = integral phi(x|theta) dG(theta), G ~ DP(alpha, G0)"})


def cheatsheet():
    return "gh_c5_1: DP mixture model: f(x) = integral phi(x|theta) dG(theta), G ~ DP(alpha, G0)"
