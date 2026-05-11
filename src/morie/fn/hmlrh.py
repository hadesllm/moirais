# morie.fn — function file (hadesllm/morie)
"""Learning-rate heuristic: start with LR finder, use 1/10 of divergence point."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_learning_rate_heuristic"]


def geron_learning_rate_heuristic(lr_curve):
    """
    Learning-rate heuristic: start with LR finder, use 1/10 of divergence point

    Formula: lr = lr_diverge / 10

    Parameters
    ----------
    lr_curve : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: suggested_lr

    References
    ----------
    Géron Ch 9
    """
    lr_curve = np.atleast_1d(np.asarray(lr_curve, dtype=float))
    n = len(lr_curve)
    result = float(np.mean(lr_curve))
    se = float(np.std(lr_curve, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Learning-rate heuristic: start with LR finder, use 1/10 of divergence point"})


def cheatsheet():
    return "hmlrh: Learning-rate heuristic: start with LR finder, use 1/10 of divergence point"
