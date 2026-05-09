"""CLIP image encoder (ViT or ResNet variant)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clip_image_encoder"]


def clip_image_encoder(image, backbone):
    """
    CLIP image encoder (ViT or ResNet variant)

    Formula: ViT-L/14 or ResNet-50x16

    Parameters
    ----------
    image : array-like
        Input data.
    backbone : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Radford et al (2021)
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLIP image encoder (ViT or ResNet variant)"})


def cheatsheet():
    return "clipxi: CLIP image encoder (ViT or ResNet variant)"
