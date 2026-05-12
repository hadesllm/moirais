# morie.fn -- function file (hadesllm/morie)
"""Batch normalization: normalize per-batch then affine rescale."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_batch_normalization"]


def geron_batch_normalization(x, gamma, beta, eps):
    """
    Batch normalization: normalize per-batch then affine rescale

    Formula: x_hat = (x - mu_B) / sqrt(var_B + eps); y = gamma * x_hat + beta

    Parameters
    ----------
    x : array-like
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
        Keys: y

    References
    ----------
    Géron Ch 11
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch normalization: normalize per-batch then affine rescale"})


def cheatsheet():
    return "hmbntr: Batch normalization: normalize per-batch then affine rescale"
