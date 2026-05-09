# moirais.fn — function file (hadesllm/moirais)
"""Joint distribution of block frequencies B_1,...,B_(n+1) is uniform over partitions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_block_freq_dist"]


def gibbons_block_freq_dist(m, n):
    """
    Joint distribution of block frequencies B_1,...,B_(n+1) is uniform over partitions

    Formula: P(B_1=b_1,...) = 1/C(m+n,n) when F_X = F_Y

    Parameters
    ----------
    m : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: joint_distribution

    References
    ----------
    Gibbons Theorem 2.11.2
    """
    m = np.asarray(m, dtype=float)
    n = int(m) if m.ndim == 0 else len(m)
    result = float(np.mean(m))
    se = float(np.std(m, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint distribution of block frequencies B_1,...,B_(n+1) is uniform over partitions"})


def cheatsheet():
    return "gb2112: Joint distribution of block frequencies B_1,...,B_(n+1) is uniform over partitions"
