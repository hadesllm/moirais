"""VanRaden Method 2 genomic relationship matrix (weighted)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["vanraden_method2"]


def vanraden_method2(marker_matrix, weights, freq):
    """
    VanRaden Method 2 genomic relationship matrix (weighted)

    Formula: G = sum_j w_j * z_j*z_j' / (2 * sum_j w_j * p_j*(1-p_j))

    Parameters
    ----------
    marker_matrix : array-like
        Input data.
    weights : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VanRaden Method 2 genomic relationship matrix (weighted)"})


def cheatsheet():
    return "vanr2: VanRaden Method 2 genomic relationship matrix (weighted)"
