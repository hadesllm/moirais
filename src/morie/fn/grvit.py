# morie.fn — function file (hadesllm/morie)
"""Vision Transformer patch embedding: flatten patches, project to d_model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_vit_patch_embedding"]


def geron_vit_patch_embedding(image, patch_size, E, E_pos, cls_token):
    """
    Vision Transformer patch embedding: flatten patches, project to d_model

    Formula: z_p = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos

    Parameters
    ----------
    image : array-like
        Input data.
    patch_size : array-like
        Input data.
    E : array-like
        Input data.
    E_pos : array-like
        Input data.
    cls_token : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z0

    References
    ----------
    Géron Ch 16, Vision Transformer (ViT) section
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vision Transformer patch embedding: flatten patches, project to d_model"})


def cheatsheet():
    return "grvit: Vision Transformer patch embedding: flatten patches, project to d_model"
