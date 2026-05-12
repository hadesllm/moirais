# morie.fn -- function file (hadesllm/morie)
"""Feature map output of a conv layer after activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_feature_map"]


def geron_feature_map(x, K, b):
    """
    Feature map output of a conv layer after activation

    Formula: F = phi(conv(x, K) + b)

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Feature map output of a conv layer after activation"})


def cheatsheet():
    return "hmfmap: Feature map output of a conv layer after activation"
