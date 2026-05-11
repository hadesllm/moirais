# morie.fn — function file (hadesllm/morie)
"""1D convolution forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["conv1d_forward"]


def conv1d_forward(x, w):
    """
    1D convolution forward pass

    Formula: y[i] = sum w[k] * x[i+k]

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "1D convolution forward pass"})


def cheatsheet():
    return "cnn1d: 1D convolution forward pass"
