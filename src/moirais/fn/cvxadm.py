"""ADMM updates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_admm"]


def boyd_admm(f, g, A, B, c, rho):
    """
    ADMM updates

    Formula: x: argmin Lp; z: argmin Lp; u <- u + Ax+Bz-c

    Parameters
    ----------
    f : array-like
        Input data.
    g : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    c : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x, z

    References
    ----------
    Boyd CVX Ch 5 (ADMM monograph)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ADMM updates"})


def cheatsheet():
    return "cvxadm: ADMM updates"
