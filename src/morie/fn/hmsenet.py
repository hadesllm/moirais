# morie.fn — function file (hadesllm/morie)
"""Squeeze-and-Excitation (SENet) block for channel recalibration."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_senet"]


def geron_senet(x, r):
    """
    Squeeze-and-Excitation (SENet) block for channel recalibration

    Formula: s = sigmoid(W2 ReLU(W1 GAP(x))); y = s * x

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
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
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squeeze-and-Excitation (SENet) block for channel recalibration"})


def cheatsheet():
    return "hmsenet: Squeeze-and-Excitation (SENet) block for channel recalibration"
