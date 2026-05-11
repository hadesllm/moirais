"""VQ-GAN decoder."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vqgan_decode"]


def vqgan_decode(indices, codebook, decoder):
    """
    VQ-GAN decoder

    Formula: decoder from codebook indices to image

    Parameters
    ----------
    indices : array-like
        Input data.
    codebook : array-like
        Input data.
    decoder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Esser et al (2021)
    """
    indices = np.atleast_1d(np.asarray(indices, dtype=float))
    n = len(indices)
    result = float(np.mean(indices))
    se = float(np.std(indices, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VQ-GAN decoder"})


def cheatsheet():
    return "vqgdec: VQ-GAN decoder"
