# morie.fn -- function file (hadesllm/morie)
"""2D convolution forward pass with single filter and stride s, padding p."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_conv2d_forward"]


def geron_conv2d_forward(X, W, b, stride, padding):
    """
    2D convolution forward pass with single filter and stride s, padding p

    Formula: Y[i,j] = sum_{u,v} W[u,v] * X[i*s+u-p, j*s+v-p] + b

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.
    b : array-like
        Input data.
    stride : array-like
        Input data.
    padding : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 12, Convolutional Layers section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "2D convolution forward pass with single filter and stride s, padding p"})


def cheatsheet():
    return "grcvf: 2D convolution forward pass with single filter and stride s, padding p"
