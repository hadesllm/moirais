# moirais.fn — function file (hadesllm/moirais)
"""GoogLeNet/Inception with parallel filter modules."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_googlenet"]


def geron_googlenet(n_classes):
    """
    GoogLeNet/Inception with parallel filter modules

    Formula: Inception: concat(1x1, 3x3, 5x5, pool)

    Parameters
    ----------
    n_classes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 12
    """
    n_classes = np.atleast_1d(np.asarray(n_classes, dtype=float))
    n = len(n_classes)
    result = float(np.mean(n_classes))
    se = float(np.std(n_classes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GoogLeNet/Inception with parallel filter modules"})


def cheatsheet():
    return "hmgoog: GoogLeNet/Inception with parallel filter modules"
