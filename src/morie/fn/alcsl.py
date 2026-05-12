# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Cosine-similarity regression loss for SBERT: MSE between predicted and target cos-sim."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_cosine_similarity_loss"]


def alammar_cosine_similarity_loss(a, b, y_true):
    """
    Cosine-similarity regression loss for SBERT: MSE between predicted and target cos-sim

    Formula: L = (cos(a, b) - y_true)^2

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    y_true : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 10, Cosine Similarity Loss section
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cosine-similarity regression loss for SBERT: MSE between predicted and target cos-sim"})


def cheatsheet():
    return "alcsl: Cosine-similarity regression loss for SBERT: MSE between predicted and target cos-sim"
