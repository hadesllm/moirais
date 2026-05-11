# morie.fn — function file (hadesllm/morie)
"""Rectified linear unit activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_relu"]


def geron_relu(z):
    """
    Rectified linear unit activation

    Formula: ReLU(z) = max(0, z)

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a

    References
    ----------
    Géron Ch 9
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rectified linear unit activation"})


def cheatsheet():
    return "hmrelu: Rectified linear unit activation"
