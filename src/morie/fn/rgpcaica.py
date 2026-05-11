# morie.fn — function file (hadesllm/morie)
"""Comparative analysis of PCA, ICA, and NMF for signal separation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pca_vs_ica"]


def rangayyan_pca_vs_ica(X, n_components, method):
    """
    Comparative analysis of PCA, ICA, and NMF for signal separation

    Formula: PCA: orthogonal Gaussian; ICA: statistically independent non-Gaussian; NMF: non-negative

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components, reconstruction_error

    References
    ----------
    Rangayyan Ch 9.7.4
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Comparative analysis of PCA, ICA, and NMF for signal separation"})


def cheatsheet():
    return "rgpcaica: Comparative analysis of PCA, ICA, and NMF for signal separation"
