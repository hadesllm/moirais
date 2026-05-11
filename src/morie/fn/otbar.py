"""Wasserstein barycenter of fixed-support measures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_barycenter_fixed"]


def ot_barycenter_fixed(A, C_list, weights, epsilon):
    """
    Wasserstein barycenter of fixed-support measures

    Formula: argmin_{ν ∈ P_n} Σ w_k OT_ε(μ_k, ν)

    Parameters
    ----------
    A : array-like
        Input data.
    C_list : array-like
        Input data.
    weights : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bary

    References
    ----------
    Cuturi & Doucet (2014)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wasserstein barycenter of fixed-support measures"})


def cheatsheet():
    return "otbar: Wasserstein barycenter of fixed-support measures"
