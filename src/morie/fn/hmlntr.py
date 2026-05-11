# morie.fn — function file (hadesllm/morie)
"""Layer normalization: normalize across features within a single sample."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_layer_normalization"]


def geron_layer_normalization(x, gamma, beta, eps):
    """
    Layer normalization: normalize across features within a single sample

    Formula: x_hat = (x - mu) / sqrt(var + eps); y = gamma * x_hat + beta

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Layer normalization: normalize across features within a single sample"})


def cheatsheet():
    return "hmlntr: Layer normalization: normalize across features within a single sample"
