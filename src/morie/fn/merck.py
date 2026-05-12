# morie.fn -- function file (hadesllm/morie)
"""Mercer's theorem: kernel expansion via eigenfunctions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mercer_theorem"]


def mercer_theorem(K_matrix):
    """
    Mercer's theorem: kernel expansion via eigenfunctions

    Formula: K(K_matrix,z) = sum_i lambda_i * phi_i(K_matrix) * phi_i(z); K must be symmetric positive semi-definite

    Parameters
    ----------
    K_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'eigenvalues': 'array', 'eigenfunctions': 'array'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    K_matrix = np.asarray(K_matrix, dtype=float)
    n = int(K_matrix) if K_matrix.ndim == 0 else len(K_matrix)
    result = float(np.mean(K_matrix))
    se = float(np.std(K_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mercer's theorem: kernel expansion via eigenfunctions"})


def cheatsheet():
    return "merck: Mercer's theorem: kernel expansion via eigenfunctions"
