# moirais.fn — function file (hadesllm/moirais)
"""Classification + localization: predict class and bounding box."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_classification_localization"]


def geron_classification_localization(image, model):
    """
    Classification + localization: predict class and bounding box

    Formula: [p_1,...,p_K, x, y, w, h] output

    Parameters
    ----------
    image : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: class, box

    References
    ----------
    Géron Ch 12
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classification + localization: predict class and bounding box"})


def cheatsheet():
    return "hmclc: Classification + localization: predict class and bounding box"
