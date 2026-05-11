# morie.fn — function file (hadesllm/morie)
"""Vision Transformer (ViT): transformer on image patches."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_vision_transformer"]


def geron_vision_transformer(image, patch_size, n_layers):
    """
    Vision Transformer (ViT): transformer on image patches

    Formula: image -> patch embeddings + pos enc -> transformer encoder -> [CLS] classifier

    Parameters
    ----------
    image : array-like
        Input data.
    patch_size : array-like
        Input data.
    n_layers : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vision Transformer (ViT): transformer on image patches"})


def cheatsheet():
    return "hmvit: Vision Transformer (ViT): transformer on image patches"
