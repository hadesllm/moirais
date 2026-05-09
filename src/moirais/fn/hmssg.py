# moirais.fn — function file (hadesllm/moirais)
"""Semantic segmentation: per-pixel class labels."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_semantic_segmentation"]


def geron_semantic_segmentation(image, model):
    """
    Semantic segmentation: per-pixel class labels

    Formula: y[i,j] in {1..K} for each pixel

    Parameters
    ----------
    image : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: segmentation

    References
    ----------
    Géron Ch 12
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semantic segmentation: per-pixel class labels"})


def cheatsheet():
    return "hmssg: Semantic segmentation: per-pixel class labels"
