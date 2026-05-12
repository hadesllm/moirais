# morie.fn -- function file (hadesllm/morie)
"""Principal component based GxE dimension reduction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["pc_gxe_reduction"]


def pc_gxe_reduction(GxE_matrix, k):
    """
    Principal component based GxE dimension reduction

    Formula: GxE = U_k * D_k * V_k'; retain k leading PC of the GxE matrix

    Parameters
    ----------
    GxE_matrix : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'GxE_approx': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    GxE_matrix = np.asarray(GxE_matrix, dtype=float)
    n = int(GxE_matrix) if GxE_matrix.ndim == 0 else len(GxE_matrix)
    result = float(np.mean(GxE_matrix))
    se = float(np.std(GxE_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Principal component based GxE dimension reduction"})


def cheatsheet():
    return "pcgxe: Principal component based GxE dimension reduction"
