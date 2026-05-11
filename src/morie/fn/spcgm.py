"""Spectrogram (STFT magnitude).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 5.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['spcgm']
def spcgm(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    nfft: int | None = None,
) -> DescriptiveResult:
    """Compute the spectrogram (STFT magnitude squared).

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Segment length.
    noverlap : int or None
        Overlap (default nperseg//2).
    window : str
        Window function name.
    nfft : int or None
        FFT length.

    Returns
    -------
    DescriptiveResult
        ``extra`` has ``frequencies``, ``times``, ``Sxx``.
    """
    from scipy.signal import spectrogram

    x = np.asarray(x, dtype=float).ravel()
    seg = min(nperseg, len(x))
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=seg, noverlap=noverlap,
                            window=window, nfft=nfft)

    return DescriptiveResult(
        name="spcgm",
        value=float(np.mean(Sxx)),
        extra={"frequencies": f, "times": t, "Sxx": Sxx},
    )


def cheatsheet() -> str:
    return "spcgm({}) -> Spectrogram (STFT)."
