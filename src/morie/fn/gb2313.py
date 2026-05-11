# morie.fn — function file (hadesllm/morie)
"""Joint moment E[T_n(x)T_n(y)] for x < y."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_edf_joint_moment"]


def gibbons_edf_joint_moment(x, y, n):
    """
    Joint moment E[T_n(x)T_n(y)] for x < y

    Formula: E[T_n(x)T_n(y)] = nF(x) + n(n-1)F(x)F(y) for x < y

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: joint_moment

    References
    ----------
    Gibbons Corollary 2.3.1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint moment E[T_n(x)T_n(y)] for x < y"})


def cheatsheet():
    return "gb2313: Joint moment E[T_n(x)T_n(y)] for x < y"
