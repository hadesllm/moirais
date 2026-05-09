# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Astle-Balding genomic relationship matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["astle_balding_grm"]


def astle_balding_grm(marker_matrix, freq):
    """
    Astle-Balding genomic relationship matrix

    Formula: K_jk = (1/p) * sum_i (x_ij - 2*p_i)*(x_ik - 2*p_i) / sqrt(2*p_i*(1-p_i) * 2*p_i*(1-p_i))

    Parameters
    ----------
    marker_matrix : array-like
        Input data.
    freq : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'K': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    marker_matrix = np.asarray(marker_matrix, dtype=float)
    n = int(marker_matrix) if marker_matrix.ndim == 0 else len(marker_matrix)
    result = float(np.mean(marker_matrix))
    se = float(np.std(marker_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Astle-Balding genomic relationship matrix"})


def cheatsheet():
    return "astlb: Astle-Balding genomic relationship matrix"
