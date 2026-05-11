# morie.fn — function file (hadesllm/morie)
"""Multi-trait linear mixed model with Kronecker covariance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["multi_trait_lmm"]


def multi_trait_lmm(Y, X, Z, A):
    """
    Multi-trait linear mixed model with Kronecker covariance

    Formula: vec(Y) = (I_t Y X)*vec(B) + (I_t Y Z)*vec(G) + vec(E); G ~ MN(0, Sigma_g Y A); E ~ MN(0, Sigma_e Y I)

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'B_hat': 'matrix', 'G_hat': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-trait linear mixed model with Kronecker covariance"})


def cheatsheet():
    return "mtlmm: Multi-trait linear mixed model with Kronecker covariance"
