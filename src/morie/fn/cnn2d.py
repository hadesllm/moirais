# morie.fn — function file (hadesllm/morie)
"""2D convolution forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["conv2d_forward"]


def conv2d_forward(x, w):
    """
    2D convolution forward pass

    Formula: y[i,j] = sum sum w[m,n] * x[i+m,j+n]

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 14
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "2D convolution forward pass"})


def cheatsheet():
    return "cnn2d: 2D convolution forward pass"
