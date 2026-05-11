# morie.fn — function file (hadesllm/morie)
"""DP conjugacy: posterior given n i.i.d. observations is DP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_post"]


def ghosal_dp_post(x):
    """
    DP conjugacy: posterior given n i.i.d. observations is DP

    Formula: G | X_1..X_n ~ DP(alpha+n, (alpha*G0 + sum delta_{X_i})/(alpha+n))

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
    Ghosal Ch 4 §4.1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP conjugacy: posterior given n i.i.d. observations is DP"})


def cheatsheet():
    return "gh_c4_6: DP conjugacy: posterior given n i.i.d. observations is DP"
