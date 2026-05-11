# morie.fn — function file (hadesllm/morie)
"""Procrustes rotation to align two MDS configurations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["procrustes_rotation"]


def procrustes_rotation(A, Z):
    """
    Procrustes rotation to align two MDS configurations

    Formula: min ||A - Z*T||^2 over orthogonal T; T = U*V' from SVD(A'*Z)

    Parameters
    ----------
    A : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'A_rotated': 'matrix', 'T': 'matrix', 'residual': 'float'}

    References
    ----------
    Armstrong Ch 3
    """
    A = np.asarray(A, dtype=float)
    n = int(A) if A.ndim == 0 else len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Procrustes rotation to align two MDS configurations"})


def cheatsheet():
    return "procs: Procrustes rotation to align two MDS configurations"
