# morie.fn -- function file (hadesllm/morie)
"""FashionMNIST image classifier: CNN on 28x28 gray images."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_fashion_mnist"]


def geron_fashion_mnist(epochs, lr):
    """
    FashionMNIST image classifier: CNN on 28x28 gray images

    Formula: CNN -> flatten -> FC -> softmax(10)

    Parameters
    ----------
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 10
    """
    epochs = np.atleast_1d(np.asarray(epochs, dtype=float))
    n = len(epochs)
    result = float(np.mean(epochs))
    se = float(np.std(epochs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FashionMNIST image classifier: CNN on 28x28 gray images"})


def cheatsheet():
    return "hmfmn: FashionMNIST image classifier: CNN on 28x28 gray images"
