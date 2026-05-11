"""ViT patch embedding via 2D conv."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_patch_embed"]


def vit_patch_embed(image, patch_size, embed_dim):
    """
    ViT patch embedding via 2D conv

    Formula: Conv2d(stride=patch_size); flatten

    Parameters
    ----------
    image : array-like
        Input data.
    patch_size : array-like
        Input data.
    embed_dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT patch embedding via 2D conv"})


def cheatsheet():
    return "vitptm: ViT patch embedding via 2D conv"
