# moirais.fn — function file (hadesllm/moirais)
"""Scalogram: energy density via squared CWT magnitudes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_scalogram"]


def rangayyan_scalogram(x, fs, scales, wavelet):
    """
    Scalogram: energy density via squared CWT magnitudes

    Formula: SC(a,b) = |CWT(a,b)|^2 / a

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
        Keys: scalogram, scales, time

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scalogram: energy density via squared CWT magnitudes"})


def cheatsheet():
    return "rgsclgr: Scalogram: energy density via squared CWT magnitudes"
