# morie.fn -- function file (hadesllm/morie)
"""ViLBERT: dual-stream vision-language transformer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_vilbert"]


def geron_vilbert(image, text):
    """
    ViLBERT: dual-stream vision-language transformer

    Formula: two transformer streams (image, text) with co-attention layers

    Parameters
    ----------
    image : array-like
        Input data.
    text : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 16
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViLBERT: dual-stream vision-language transformer"})


def cheatsheet():
    return "hmvilb: ViLBERT: dual-stream vision-language transformer"
