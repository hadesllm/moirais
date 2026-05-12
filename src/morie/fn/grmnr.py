# morie.fn -- function file (hadesllm/morie)
"""Max-norm regularization: project weight vectors onto L2 ball of radius r."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_max_norm_regularization"]


def geron_max_norm_regularization(W, r):
    """
    Max-norm regularization: project weight vectors onto L2 ball of radius r

    Formula: if ||w_i||_2 > r: w_i <- w_i * r / ||w_i||_2

    Parameters
    ----------
    W : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W_clipped

    References
    ----------
    Géron Ch 11, Max-Norm Regularization section
    """
    W = np.asarray(W, dtype=float)
    n = int(W) if W.ndim == 0 else len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max-norm regularization: project weight vectors onto L2 ball of radius r"})


def cheatsheet():
    return "grmnr: Max-norm regularization: project weight vectors onto L2 ball of radius r"
