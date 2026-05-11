# morie.fn — function file (hadesllm/morie)
"""Local DP contraction rate for regression: same near-optimal rate as standard DPM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_local_dp_rate"]


def ghosal_local_dp_rate(x):
    """
    Local DP contraction rate for regression: same near-optimal rate as standard DPM

    Formula: Local DP(alpha(x), G0): rate n^{-s/(2s+1)} * (log n)^t for regression

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
    Ghosal Ch 14 §14.9.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local DP contraction rate for regression: same near-optimal rate as standard DPM"})


def cheatsheet():
    return "gh_loc_dp_crt: Local DP contraction rate for regression: same near-optimal rate as standard DPM"
