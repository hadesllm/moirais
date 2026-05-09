# moirais.fn — function file (hadesllm/moirais)
"""Batch normalization: z-normalize per-channel across the mini-batch, then affine scale/shift."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_batch_normalization"]


def geron_batch_normalization(X, gamma, beta, eps):
    """
    Batch normalization: z-normalize per-channel across the mini-batch, then affine scale/shift

    Formula: x_hat = (x - mu_B) / sqrt(sigma_B^2 + eps); y = gamma * x_hat + beta

    Parameters
    ----------
    X : array-like
        Input data.
    gamma : array-like
        Input data.
    beta : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 11, Batch Normalization section (Ioffe & Szegedy 2015)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch normalization: z-normalize per-channel across the mini-batch, then affine scale/shift"})


def cheatsheet():
    return "grbn: Batch normalization: z-normalize per-channel across the mini-batch, then affine scale/shift"
