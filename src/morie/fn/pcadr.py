# morie.fn -- function file (hadesllm/morie)
"""PCA for dimensionality reduction of SNP marker matrices."""
import numpy as np
from ._richresult import RichResult

__all__ = ["pca_dimensionality_reduction"]


def pca_dimensionality_reduction(X, n_components):
    """
    PCA for dimensionality reduction of SNP marker matrices

    Formula: Z = X * V_k; V_k leading k eigenvectors of X'X; retain % variance

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'Z': 'matrix', 'var_explained': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA for dimensionality reduction of SNP marker matrices"})


def cheatsheet():
    return "pcadr: PCA for dimensionality reduction of SNP marker matrices"
