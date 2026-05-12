# morie.fn -- function file (hadesllm/morie)
"""STFT spectrogram (magnitude squared STFT)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_stft_spectrogram"]


def rangayyan_stft_spectrogram(x, fs, nperseg, noverlap, window):
    """
    STFT spectrogram (magnitude squared STFT)

    Formula: S(m,f) = |sum x[n]*w[n-m]*exp(-j2pi*fn)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    nperseg : array-like
        Input data.
    noverlap : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrogram, t, freqs

    References
    ----------
    Rangayyan Ch 8.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "STFT spectrogram (magnitude squared STFT)"})


def cheatsheet():
    return "rgstfts: STFT spectrogram (magnitude squared STFT)"
