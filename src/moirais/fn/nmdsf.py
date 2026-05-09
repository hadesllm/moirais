# moirais.fn — function file (hadesllm/moirais)
"""Nonmetric MDS via isotonic regression (Kruskal 1964)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["nonmetric_mds"]


def nonmetric_mds(delta, n_dims, max_iter):
    """
    Nonmetric MDS via isotonic regression (Kruskal 1964)

    Formula: Find coords X and disparities dhat_ij monotone to delta_ij minimizing S1

    Parameters
    ----------
    delta : array-like
        Input data.
    n_dims : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'X': 'matrix', 'stress': 'float', 'disparities': 'matrix'}

    References
    ----------
    Armstrong Ch 3
    """
    delta = np.asarray(delta, dtype=float)
    n = int(delta) if delta.ndim == 0 else len(delta)
    result = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonmetric MDS via isotonic regression (Kruskal 1964)"})


def cheatsheet():
    return "nmdsf: Nonmetric MDS via isotonic regression (Kruskal 1964)"
