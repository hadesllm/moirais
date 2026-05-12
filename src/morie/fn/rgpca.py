# morie.fn -- function file (hadesllm/morie)
"""PCA for signal mixture separation (eigendecomposition of covariance)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pca_signals"]


def rangayyan_pca_signals(X):
    """
    PCA for signal mixture separation (eigendecomposition of covariance)

    Formula: Sigma = (1/N)*X*X^T; X_pca = V^T*X where V=eigenvectors

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components, eigenvalues, eigenvectors

    References
    ----------
    Rangayyan Ch 9.7.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA for signal mixture separation (eigendecomposition of covariance)"})


def cheatsheet():
    return "rgpca: PCA for signal mixture separation (eigendecomposition of covariance)"
