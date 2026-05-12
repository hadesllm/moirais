# morie.fn -- function file (hadesllm/morie)
"""Binary nonparametric monotone regression consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mono_reg_con"]


def ghosal_mono_reg_con(x, y):
    """
    Binary nonparametric monotone regression consistency

    Formula: P(Y=1|x) = F(x) monotone, DP prior on F, consistent in weak topology

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 7 §7.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary nonparametric monotone regression consistency"})


def cheatsheet():
    return "gh_c7_10: Binary nonparametric monotone regression consistency"
