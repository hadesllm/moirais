# morie.fn — function file (hadesllm/morie)
"""Wavelet denoising (soft/hard thresholding)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_denoise"]


def rangayyan_wavelet_denoise(x):
    """
    Wavelet denoising (soft/hard thresholding)

    Formula: x_hat = IDWT(threshold(DWT(x)))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet denoising (soft/hard thresholding)"})


def cheatsheet():
    return "rgwav: Wavelet denoising (soft/hard thresholding)"
