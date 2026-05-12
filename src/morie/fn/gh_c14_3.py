# morie.fn -- function file (hadesllm/morie)
"""Chinese restaurant process: seat customer n+1 at table k prop n_k/(alpha+n) or new."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_crp_def"]


def ghosal_crp_def(x):
    """
    Chinese restaurant process: seat customer n+1 at table k prop n_k/(alpha+n) or new

    Formula: P(C_{n+1}=k | C_1..C_n) = n_k/(alpha+n) or alpha/(alpha+n) new table

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
    Ghosal Ch 14 §14.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chinese restaurant process: seat customer n+1 at table k prop n_k/(alpha+n) or new"})


def cheatsheet():
    return "gh_c14_3: Chinese restaurant process: seat customer n+1 at table k prop n_k/(alpha+n) or new"
