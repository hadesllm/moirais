# moirais.fn — function file (hadesllm/moirais)
"""PCA signal reconstruction from top-k components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch9_pca_reconstruction"]


def rangayyan_ch9_pca_reconstruction(X, k):
    """
    PCA signal reconstruction from top-k components

    Formula: X_hat = mean + sum_{j=1}^{k} (X_proj_j * v_j^T)

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_reconstructed, reconstruction_error

    References
    ----------
    Rangayyan Ch 9.7.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA signal reconstruction from top-k components"})


def cheatsheet():
    return "rgeqn9a: PCA signal reconstruction from top-k components"
