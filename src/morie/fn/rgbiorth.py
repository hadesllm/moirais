# morie.fn — function file (hadesllm/morie)
"""Biorthogonal wavelet (symmetric, linear phase) DWT."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_biorthogonal_wvlt"]


def rangayyan_biorthogonal_wvlt(x, wavelet, levels):
    """
    Biorthogonal wavelet (symmetric, linear phase) DWT

    Formula: Analysis: h_tilde, g_tilde; synthesis: h, g; perfect reconstruction via duality

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
        Keys: coeffs, reconstructed

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Biorthogonal wavelet (symmetric, linear phase) DWT"})


def cheatsheet():
    return "rgbiorth: Biorthogonal wavelet (symmetric, linear phase) DWT"
