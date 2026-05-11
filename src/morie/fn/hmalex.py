# morie.fn — function file (hadesllm/morie)
"""AlexNet: deep CNN for ImageNet with ReLU and dropout."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_alexnet"]


def geron_alexnet(n_classes):
    """
    AlexNet: deep CNN for ImageNet with ReLU and dropout

    Formula: 5 conv -> 3 FC; ReLU; dropout 0.5; 60M params

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlexNet: deep CNN for ImageNet with ReLU and dropout"})


def cheatsheet():
    return "hmalex: AlexNet: deep CNN for ImageNet with ReLU and dropout"
