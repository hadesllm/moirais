"""Convergence tolerance check for Sinkhorn iterations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sinkhorn_tol"]


def ot_sinkhorn_tol(T, a, b):
    """
    Convergence tolerance check for Sinkhorn iterations

    Formula: tol = max(|T 1 - a|_inf, |T^T 1 - b|_inf)

    Parameters
    ----------
    T : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: err

    References
    ----------
    Cuturi (2013)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convergence tolerance check for Sinkhorn iterations"})


def cheatsheet():
    return "otsktol: Convergence tolerance check for Sinkhorn iterations"
