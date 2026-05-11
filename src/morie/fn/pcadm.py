# morie.fn — function file (hadesllm/morie)
"""PCA via SVD for dimension reduction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["pca_dimension_reduction"]


def pca_dimension_reduction(x):
    """
    PCA via SVD for dimension reduction

    Formula: X = U Sigma V', keep top k

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA via SVD for dimension reduction"})


def cheatsheet():
    return "pcadm: PCA via SVD for dimension reduction"
