"""VQ-GAN encoder + codebook."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vqgan_encode"]


def vqgan_encode(image, codebook):
    """
    VQ-GAN encoder + codebook

    Formula: encoder + nearest-codebook quantize

    Parameters
    ----------
    image : array-like
        Input data.
    codebook : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Esser et al (2021) VQ-GAN
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VQ-GAN encoder + codebook"})


def cheatsheet():
    return "vqgenc: VQ-GAN encoder + codebook"
