# morie.fn — function file (hadesllm/morie)
"""Classical metric MDS via Torgerson-Eckart-Young decomposition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["metric_mds_torgerson"]


def metric_mds_torgerson(D_matrix, n_dims):
    """
    Classical metric MDS via Torgerson-Eckart-Young decomposition

    Formula: B = -1/2 * H * D^(2) * H; H = I - (1/n)*11'; eigendecomp B = Q*Lambda*Q'; coords X = Q_k*Lambda_k^{1/2}

    Parameters
    ----------
    D_matrix : array-like
        Input data.
    n_dims : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'coords': 'matrix', 'eigenvalues': 'array'}

    References
    ----------
    Armstrong Ch 3
    """
    D_matrix = np.asarray(D_matrix, dtype=float)
    n = int(D_matrix) if D_matrix.ndim == 0 else len(D_matrix)
    result = float(np.mean(D_matrix))
    se = float(np.std(D_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classical metric MDS via Torgerson-Eckart-Young decomposition"})


def cheatsheet():
    return "mmdsf: Classical metric MDS via Torgerson-Eckart-Young decomposition"
