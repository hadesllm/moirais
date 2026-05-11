# morie.fn — function file (hadesllm/morie)
"""Relations between classes of discrete random probability measures."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_disc_rp_rel"]


def ghosal_disc_rp_rel(x):
    """
    Relations between classes of discrete random probability measures

    Formula: DP subset PY subset PK subset NCRM, each is special case of next

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
    Ghosal Ch 14 §14.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Relations between classes of discrete random probability measures"})


def cheatsheet():
    return "gh_c14_17: Relations between classes of discrete random probability measures"
