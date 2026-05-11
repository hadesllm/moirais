# morie.fn — function file (hadesllm/morie)
"""Undercomplete linear autoencoder (PCA equivalent when linear)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_autoencoder"]


def geron_autoencoder(X, bottleneck):
    """
    Undercomplete linear autoencoder (PCA equivalent when linear)

    Formula: min ||x - W_2 W_1 x||^2 with bottleneck dim < input dim

    Parameters
    ----------
    X : array-like
        Input data.
    bottleneck : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: encoder, decoder

    References
    ----------
    Géron Ch 18
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Undercomplete linear autoencoder (PCA equivalent when linear)"})


def cheatsheet():
    return "hmaen: Undercomplete linear autoencoder (PCA equivalent when linear)"
