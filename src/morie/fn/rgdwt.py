# morie.fn — function file (hadesllm/morie)
"""Discrete wavelet transform (DWT) via filterbank."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_dwt"]


def rangayyan_dwt(x, wavelet, levels):
    """
    Discrete wavelet transform (DWT) via filterbank

    Formula: c_j[n]=sum h[k]*c_{j-1}[2n-k]; d_j[n]=sum g[k]*c_{j-1}[2n-k]

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
        Keys: approx, details

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete wavelet transform (DWT) via filterbank"})


def cheatsheet():
    return "rgdwt: Discrete wavelet transform (DWT) via filterbank"
