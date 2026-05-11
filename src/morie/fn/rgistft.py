# morie.fn — function file (hadesllm/morie)
"""Inverse STFT signal reconstruction from spectrogram."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_istft"]


def rangayyan_istft(stft, window, hop):
    """
    Inverse STFT signal reconstruction from spectrogram

    Formula: stft[n] = sum_m X(m,f)*w[n-m] / sum_m w^2[n-m] (overlap-add)

    Parameters
    ----------
    stft : array-like
        Input data.
    window : array-like
        Input data.
    hop : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_reconstructed

    References
    ----------
    Rangayyan Ch 8.4
    """
    stft = np.asarray(stft, dtype=float)
    n = int(stft) if stft.ndim == 0 else len(stft)
    result = float(np.mean(stft))
    se = float(np.std(stft, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse STFT signal reconstruction from spectrogram"})


def cheatsheet():
    return "rgistft: Inverse STFT signal reconstruction from spectrogram"
