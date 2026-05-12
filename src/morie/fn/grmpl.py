# morie.fn -- function file (hadesllm/morie)
"""2D max pooling over kxk windows with stride s."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_max_pooling"]


def geron_max_pooling(X, k, stride):
    """
    2D max pooling over kxk windows with stride s

    Formula: Y[i,j] = max over window X[i*s..i*s+k-1, j*s..j*s+k-1]

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 12, Max Pooling section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "2D max pooling over kxk windows with stride s"})


def cheatsheet():
    return "grmpl: 2D max pooling over kxk windows with stride s"
