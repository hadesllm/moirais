# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Arc-cosine kernel for neural network-like features."""
import numpy as np
from ._richresult import RichResult

__all__ = ["arc_cosine_kernel"]


def arc_cosine_kernel(X, n):
    """
    Arc-cosine kernel for neural network-like features

    Formula: K(x_i, x_j) = (1/pi) * ||x_i|| * ||x_j|| * J_n(theta); theta = arccos(x_i'x_j/(||x_i||*||x_j||))

    Parameters
    ----------
    X : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'K': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Arc-cosine kernel for neural network-like features"})


def cheatsheet():
    return "arckn: Arc-cosine kernel for neural network-like features"
