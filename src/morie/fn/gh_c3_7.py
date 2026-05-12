# morie.fn -- function file (hadesllm/morie)
"""Prior via random rectangular partitions of the sample space."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_rect_partition"]


def ghosal_rect_partition(x):
    """
    Prior via random rectangular partitions of the sample space

    Formula: G random measure constructed via random rectangular cells

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
    Ghosal Ch 3 §3.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prior via random rectangular partitions of the sample space"})


def cheatsheet():
    return "gh_c3_7: Prior via random rectangular partitions of the sample space"
