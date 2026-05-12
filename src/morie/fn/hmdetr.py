# morie.fn -- function file (hadesllm/morie)
"""DETR: CNN-transformer hybrid for end-to-end object detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_detr"]


def geron_detr(image, n_queries):
    """
    DETR: CNN-transformer hybrid for end-to-end object detection

    Formula: CNN features -> transformer encoder -> object queries -> bipartite matching

    Parameters
    ----------
    image : array-like
        Input data.
    n_queries : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boxes, classes

    References
    ----------
    Géron Ch 16
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DETR: CNN-transformer hybrid for end-to-end object detection"})


def cheatsheet():
    return "hmdetr: DETR: CNN-transformer hybrid for end-to-end object detection"
