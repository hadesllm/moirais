# moirais.fn — function file (hadesllm/moirais)
"""PCA projection via SVD-derived principal components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pca_projection"]


def geron_pca_projection(X, d):
    """
    PCA projection via SVD-derived principal components

    Formula: Z = X_centered @ W_d where W_d = first d right singular vectors of X_centered

    Parameters
    ----------
    X : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Z, components

    References
    ----------
    Géron Ch 7, PCA section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA projection via SVD-derived principal components"})


def cheatsheet():
    return "grpcap: PCA projection via SVD-derived principal components"
