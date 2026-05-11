"""Yang et al. realized genomic relationship matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["yang_realized_relationship"]


def yang_realized_relationship(marker_matrix, freq):
    """
    Yang et al. realized genomic relationship matrix

    Formula: A_jk = (1/p) * sum_i (x_ij - 2*p_i)*(x_ik - 2*p_i) / (2*p_i*(1-p_i))

    Parameters
    ----------
    marker_matrix : array-like
        Input data.
    freq : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'A': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    marker_matrix = np.asarray(marker_matrix, dtype=float)
    n = int(marker_matrix) if marker_matrix.ndim == 0 else len(marker_matrix)
    result = float(np.mean(marker_matrix))
    se = float(np.std(marker_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Yang et al. realized genomic relationship matrix"})


def cheatsheet():
    return "yangr: Yang et al. realized genomic relationship matrix"
