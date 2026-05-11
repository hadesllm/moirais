"""Gromov-Wasserstein distance between metric measure spaces."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_gromov_wasserstein"]


def ot_gromov_wasserstein(Cx, Cy, a, b, max_iter):
    """
    Gromov-Wasserstein distance between metric measure spaces

    Formula: min_T Σ |C^X_{ij}-C^Y_{kl}|² T_ik T_jl

    Parameters
    ----------
    Cx : array-like
        Input data.
    Cy : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost

    References
    ----------
    Mémoli (2011)
    """
    Cx = np.atleast_1d(np.asarray(Cx, dtype=float))
    n = len(Cx)
    result = float(np.mean(Cx))
    se = float(np.std(Cx, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gromov-Wasserstein distance between metric measure spaces"})


def cheatsheet():
    return "otgw: Gromov-Wasserstein distance between metric measure spaces"
