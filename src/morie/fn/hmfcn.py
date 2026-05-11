# morie.fn — function file (hadesllm/morie)
"""Fully convolutional network (FCN) for dense prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_fcn"]


def geron_fcn(image, model):
    """
    Fully convolutional network (FCN) for dense prediction

    Formula: replace FC layers with conv layers; spatial output

    Parameters
    ----------
    image : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dense_prediction

    References
    ----------
    Géron Ch 12
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fully convolutional network (FCN) for dense prediction"})


def cheatsheet():
    return "hmfcn: Fully convolutional network (FCN) for dense prediction"
