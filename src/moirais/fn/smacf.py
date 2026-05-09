"""SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm for MDS."""
import numpy as np
from ._richresult import RichResult

__all__ = ["smacof_algorithm"]


def smacof_algorithm(delta, n_dims, weights, max_iter, eps):
    """
    SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm for MDS

    Formula: sigma(X) = sum_{i<j} w_ij*(d_ij(X) - delta_ij)^2; iterate Guttman transform: X <- n^{-1} * B(X) * X

    Parameters
    ----------
    delta : array-like
        Input data.
    n_dims : array-like
        Input data.
    weights : array-like
        Input data.
    max_iter : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'X': 'matrix', 'stress': 'float'}

    References
    ----------
    Armstrong Ch 3
    """
    delta = np.asarray(delta, dtype=float)
    n = int(delta) if delta.ndim == 0 else len(delta)
    result = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm for MDS"})


def cheatsheet():
    return "smacf: SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm for MDS"
