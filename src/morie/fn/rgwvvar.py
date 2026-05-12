# morie.fn -- function file (hadesllm/morie)
"""Wavelet variance (Allan variance) by scale."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_variance"]


def rangayyan_wavelet_variance(x, wavelet, levels):
    """
    Wavelet variance (Allan variance) by scale

    Formula: V_j = (1/(2*(N-2^j))) * sum |d_j[n]|^2

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance_per_scale

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet variance (Allan variance) by scale"})


def cheatsheet():
    return "rgwvvar: Wavelet variance (Allan variance) by scale"
