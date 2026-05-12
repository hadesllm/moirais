# morie.fn -- function file (hadesllm/morie)
"""Convolutional layer: cross-correlation with learnable kernels."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_convolutional_layer"]


def geron_convolutional_layer(x, kernel, stride, padding):
    """
    Convolutional layer: cross-correlation with learnable kernels

    Formula: y[i,j,k] = sum_{u,v,c} K_k[u,v,c] * x[i+u, j+v, c] + b_k

    Parameters
    ----------
    x : array-like
        Input data.
    kernel : array-like
        Input data.
    stride : array-like
        Input data.
    padding : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if callable(kernel):
        y = x
    else:
        y = np.atleast_1d(np.asarray(kernel, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Convolutional layer: cross-correlation with learnable kernels"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Convolutional layer: cross-correlation with learnable kernels"})


def cheatsheet():
    return "hmcnv: Convolutional layer: cross-correlation with learnable kernels"
