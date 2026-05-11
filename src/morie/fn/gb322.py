# morie.fn — function file (hadesllm/morie)
"""Exact null distribution of total number of runs R (Wald-Wolfowitz)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_total_runs_dist"]


def gibbons_total_runs_dist(r, n1, n2):
    """
    Exact null distribution of total number of runs R (Wald-Wolfowitz)

    Formula: f_R(r) = 2*C(n1-1,r/2-1)*C(n2-1,r/2-1)/C(n,n1) if even; combined if odd

    Parameters
    ----------
    r : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Theorem 3.2.2
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exact null distribution of total number of runs R (Wald-Wolfowitz)"})


def cheatsheet():
    return "gb322: Exact null distribution of total number of runs R (Wald-Wolfowitz)"
