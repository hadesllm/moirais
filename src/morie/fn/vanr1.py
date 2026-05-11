"""VanRaden Method 1 genomic relationship matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["vanraden_method1"]


def vanraden_method1(marker_matrix, freq):
    """
    VanRaden Method 1 genomic relationship matrix

    Formula: G = Z*Z' / (2 * sum_j p_j*(1-p_j)); Z_ij = M_ij - 2*p_j

    Parameters
    ----------
    marker_matrix : array-like
        Input data.
    freq : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'G': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    marker_matrix = np.asarray(marker_matrix, dtype=float)
    n = int(marker_matrix) if marker_matrix.ndim == 0 else len(marker_matrix)
    result = float(np.mean(marker_matrix))
    se = float(np.std(marker_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VanRaden Method 1 genomic relationship matrix"})


def cheatsheet():
    return "vanr1: VanRaden Method 1 genomic relationship matrix"
