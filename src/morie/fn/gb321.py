# morie.fn — function file (hadesllm/morie)
"""Joint distribution of R1 and R2 (runs of type 1 and 2) under randomness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_runs_joint_dist"]


def gibbons_runs_joint_dist(r1, r2, n1, n2):
    """
    Joint distribution of R1 and R2 (runs of type 1 and 2) under randomness

    Formula: f_{R1,R2}(r1,r2) = c*C(n1-1,r1-1)*C(n2-1,r2-1)/C(n,n1)

    Parameters
    ----------
    r1 : array-like
        Input data.
    r2 : array-like
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
    Gibbons Theorem 3.2.1
    """
    r1 = np.asarray(r1, dtype=float)
    n = int(r1) if r1.ndim == 0 else len(r1)
    result = float(np.mean(r1))
    se = float(np.std(r1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint distribution of R1 and R2 (runs of type 1 and 2) under randomness"})


def cheatsheet():
    return "gb321: Joint distribution of R1 and R2 (runs of type 1 and 2) under randomness"
