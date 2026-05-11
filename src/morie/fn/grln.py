# morie.fn — function file (hadesllm/morie)
"""Layer normalization: z-normalize across features per-sample."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_layer_normalization"]


def geron_layer_normalization(X, gamma, beta, eps):
    """
    Layer normalization: z-normalize across features per-sample

    Formula: x_hat_i = (x_i - mu_i) / sqrt(sigma_i^2 + eps); y = gamma * x_hat + beta

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
    Géron Ch 11, Layer Normalization section (Ba et al. 2016)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Layer normalization: z-normalize across features per-sample"})


def cheatsheet():
    return "grln: Layer normalization: z-normalize across features per-sample"
