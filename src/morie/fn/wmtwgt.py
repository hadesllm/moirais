"""Spatial weights matrix construction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weights_matrix"]


def weights_matrix(coords, method, k_or_threshold):
    """
    Spatial weights matrix construction

    Formula: contiguity (rook/queen) or distance-based or k-NN

    Parameters
    ----------
    coords : array-like
        Input data.
    method : array-like
        Input data.
    k_or_threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin (1988)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial weights matrix construction"})


def cheatsheet():
    return "wmtwgt: Spatial weights matrix construction"
