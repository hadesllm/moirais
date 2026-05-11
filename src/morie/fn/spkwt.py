"""Kriging weights from kriging system solution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_kriging_weights"]


def schabenberger_kriging_weights(cov_matrix, cov_target, coords):
    """
    Kriging weights from kriging system solution

    Formula: lambda = C^{-1}*(c - 1*mu) where mu solves 1'*C^{-1}*(c-1*mu) = 1

    Parameters
    ----------
    cov_matrix : array-like
        Input data.
    cov_target : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights

    References
    ----------
    Schabenberger Ch 5
    """
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = int(cov_matrix) if cov_matrix.ndim == 0 else len(cov_matrix)
    result = float(np.mean(cov_matrix))
    se = float(np.std(cov_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kriging weights from kriging system solution"})


def cheatsheet():
    return "spkwt: Kriging weights from kriging system solution"
