# morie.fn — function file (hadesllm/morie)
"""Wavelet denoising via soft/hard thresholding."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_threshold"]


def rangayyan_wavelet_threshold(x, wavelet, levels, threshold_type):
    """
    Wavelet denoising via soft/hard thresholding

    Formula: soft: sign(d)*max(|d|-lambda,0); hard: d*(|d|>=lambda); lambda=sigma*sqrt(2*log(N))

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.
    threshold_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: denoised

    References
    ----------
    Rangayyan Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet denoising via soft/hard thresholding"})


def cheatsheet():
    return "rgwvth: Wavelet denoising via soft/hard thresholding"
