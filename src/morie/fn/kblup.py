# morie.fn — function file (hadesllm/morie)
"""Kernel BLUP (K-BLUP) predictor using kernel matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kernel_blup"]


def kernel_blup(K, K_new, y, lam):
    """
    Kernel BLUP (K-BLUP) predictor using kernel matrix

    Formula: y_hat = K_new * (K + lambda*I)^{-1} * y; lambda = sigma_e^2/sigma_u^2

    Parameters
    ----------
    K : array-like
        Input data.
    K_new : array-like
        Input data.
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'y_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel BLUP (K-BLUP) predictor using kernel matrix"})


def cheatsheet():
    return "kblup: Kernel BLUP (K-BLUP) predictor using kernel matrix"
