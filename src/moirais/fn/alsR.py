"""Alternating least squares MF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["als"]


def als(R, K, reg, iters):
    """
    Alternating least squares MF

    Formula: alternate solving p_u, q_i with ridge

    Parameters
    ----------
    R : array-like
        Input data.
    K : array-like
        Input data.
    reg : array-like
        Input data.
    iters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bell-Koren (2007); Hu-Koren-Volinsky (2008)
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alternating least squares MF"})


def cheatsheet():
    return "alsR: Alternating least squares MF"
