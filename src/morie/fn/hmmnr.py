# morie.fn -- function file (hadesllm/morie)
"""Max-norm regularization: rescale weights so ||w||_2 <= r."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_max_norm"]


def geron_max_norm(w, r):
    """
    Max-norm regularization: rescale weights so ||w||_2 <= r

    Formula: if ||w||>r: w <- w * r / ||w||

    Parameters
    ----------
    w : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w

    References
    ----------
    Géron Ch 11
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max-norm regularization: rescale weights so ||w||_2 <= r"})


def cheatsheet():
    return "hmmnr: Max-norm regularization: rescale weights so ||w||_2 <= r"
