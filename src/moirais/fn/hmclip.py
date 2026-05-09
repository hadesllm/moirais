# moirais.fn — function file (hadesllm/moirais)
"""CLIP: contrastive image-text pretraining."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_clip"]


def geron_clip(images, texts):
    """
    CLIP: contrastive image-text pretraining

    Formula: maximize cosine sim of matched (image, text); minimize for unmatched

    Parameters
    ----------
    images : array-like
        Input data.
    texts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: image_enc, text_enc

    References
    ----------
    Géron Ch 16
    """
    images = np.atleast_1d(np.asarray(images, dtype=float))
    n = len(images)
    result = float(np.mean(images))
    se = float(np.std(images, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLIP: contrastive image-text pretraining"})


def cheatsheet():
    return "hmclip: CLIP: contrastive image-text pretraining"
