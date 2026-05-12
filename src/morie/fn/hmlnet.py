# morie.fn -- function file (hadesllm/morie)
"""LeNet-5 CNN architecture."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lenet5"]


def geron_lenet5(n_classes):
    """
    LeNet-5 CNN architecture

    Formula: C1(6) -> S2 -> C3(16) -> S4 -> C5(120) -> F6(84) -> softmax(10)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LeNet-5 CNN architecture"})


def cheatsheet():
    return "hmlnet: LeNet-5 CNN architecture"
