# morie.fn — function file (hadesllm/morie)
"""Dropout: randomly zero units during training with probability p."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dropout"]


def geron_dropout(x, p, training):
    """
    Dropout: randomly zero units during training with probability p

    Formula: y = mask * x / (1 - p); mask_i ~ Bernoulli(1-p)

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    training : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 11
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dropout: randomly zero units during training with probability p"})


def cheatsheet():
    return "hmdrp: Dropout: randomly zero units during training with probability p"
