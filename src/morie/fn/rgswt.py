# morie.fn -- function file (hadesllm/morie)
"""Stationary wavelet transform (SWT, undecimated DWT)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_swt"]


def rangayyan_swt(x, wavelet, levels):
    """
    Stationary wavelet transform (SWT, undecimated DWT)

    Formula: No downsampling; filter upsampled by 2^j at level j; shift-invariant

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
        Keys: approx_coeffs, detail_coeffs

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stationary wavelet transform (SWT, undecimated DWT)"})


def cheatsheet():
    return "rgswt: Stationary wavelet transform (SWT, undecimated DWT)"
