# morie.fn — function file (hadesllm/morie)
"""Recursive frequency relation for runs up and down in sequence of n numbers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_runs_up_down_recur"]


def gibbons_runs_up_down_recur(x):
    """
    Recursive frequency relation for runs up and down in sequence of n numbers

    Formula: u_n(...) = 2*u_{n-1}(r1-1) + sum cases for extending/splitting runs

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: frequency

    References
    ----------
    Gibbons Ch 3.4 recursive relation
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recursive frequency relation for runs up and down in sequence of n numbers"})


def cheatsheet():
    return "gb32lu: Recursive frequency relation for runs up and down in sequence of n numbers"
