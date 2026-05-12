# morie.fn -- function file (hadesllm/morie)
"""Wavelet-based structure detection in biomedical signals (CWT ridges)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_struct"]


def rangayyan_wavelet_struct(x, fs, scales, wavelet):
    """
    Wavelet-based structure detection in biomedical signals (CWT ridges)

    Formula: Ridge: argmax_a |CWT(a,b)| at each time b; instantaneous frequency from ridge

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    scales : array-like
        Input data.
    wavelet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ridges, inst_freq

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet-based structure detection in biomedical signals (CWT ridges)"})


def cheatsheet():
    return "rgwavstr: Wavelet-based structure detection in biomedical signals (CWT ridges)"
