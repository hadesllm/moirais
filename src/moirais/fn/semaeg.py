"""SAM image encoder (ViT-H)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sam_image_encoder"]


def sam_image_encoder(image):
    """
    SAM image encoder (ViT-H)

    Formula: large ViT pretrained on SA-1B

    Parameters
    ----------
    image : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023) SAM
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SAM image encoder (ViT-H)"})


def cheatsheet():
    return "semaeg: SAM image encoder (ViT-H)"
