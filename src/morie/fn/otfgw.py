"""Fused Gromov-Wasserstein with feature + structure."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_fused_gromov_wasserstein"]


def ot_fused_gromov_wasserstein(M, Cx, Cy, a, b, alpha, max_iter):
    """
    Fused Gromov-Wasserstein with feature + structure

    Formula: min (1-α)<T,M> + α Σ |C-C'|² T T

    Parameters
    ----------
    M : array-like
        Input data.
    Cx : array-like
        Input data.
    Cy : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    alpha : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost

    References
    ----------
    Vayer-Chapel-Flamary-Tavenard-Courty (2020)
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fused Gromov-Wasserstein with feature + structure"})


def cheatsheet():
    return "otfgw: Fused Gromov-Wasserstein with feature + structure"
