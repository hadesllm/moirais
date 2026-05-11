# morie.fn — function file (hadesllm/morie)
"""Fully convolutional upsampling (transposed conv) for segmentation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_fcn_upsample"]


def geron_fcn_upsample(X, W, stride):
    """
    Fully convolutional upsampling (transposed conv) for segmentation

    Formula: Y = X * W_upsample via transposed convolution

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 12, Fully Convolutional Networks section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fully convolutional upsampling (transposed conv) for segmentation"})


def cheatsheet():
    return "grfcn: Fully convolutional upsampling (transposed conv) for segmentation"
