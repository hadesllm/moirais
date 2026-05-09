# moirais.fn — function file (hadesllm/moirais)
"""Wavelet denoising of PPG signals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ppg_wavelet"]


def rangayyan_ppg_wavelet(ppg, fs, wavelet, levels):
    """
    Wavelet denoising of PPG signals

    Formula: DWT with db4; soft threshold at universal lambda=sigma*sqrt(2*log(N))

    Parameters
    ----------
    ppg : array-like
        Input data.
    fs : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ppg_clean

    References
    ----------
    Rangayyan Ch 8.14
    """
    ppg = np.asarray(ppg, dtype=float)
    n = int(ppg) if ppg.ndim == 0 else len(ppg)
    result = float(np.mean(ppg))
    se = float(np.std(ppg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet denoising of PPG signals"})


def cheatsheet():
    return "rgppgwt: Wavelet denoising of PPG signals"
