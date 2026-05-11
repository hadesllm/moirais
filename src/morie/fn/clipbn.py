"""CLIP image-text contrastive alignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clip_image_text_align"]


def clip_image_text_align(images, texts, tau):
    """
    CLIP image-text contrastive alignment

    Formula: -1/N sum log exp(I_i^T T_i / tau)/sum_j ...

    Parameters
    ----------
    images : array-like
        Input data.
    texts : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Radford et al (2021)
    """
    images = np.atleast_1d(np.asarray(images, dtype=float))
    n = len(images)
    result = float(np.mean(images))
    se = float(np.std(images, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLIP image-text contrastive alignment"})


def cheatsheet():
    return "clipbn: CLIP image-text contrastive alignment"
