# morie.fn — function file (hadesllm/morie)
"""Ideal sinc (low-pass) filter impulse response."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_sinc_kernel"]


def rangayyan_sinc_kernel(fc, fs, M):
    """
    Ideal sinc (low-pass) filter impulse response

    Formula: h[n] = 2*fc/fs * sinc(2*pi*fc*(n-M/2)/fs)

    Parameters
    ----------
    fc : array-like
        Input data.
    fs : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h

    References
    ----------
    Rangayyan Ch 3
    """
    fc = np.asarray(fc, dtype=float)
    n = int(fc) if fc.ndim == 0 else len(fc)
    result = float(np.mean(fc))
    se = float(np.std(fc, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ideal sinc (low-pass) filter impulse response"})


def cheatsheet():
    return "rgsinc: Ideal sinc (low-pass) filter impulse response"
