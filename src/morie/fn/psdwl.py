# morie.fn -- function file (hadesllm/morie)
"""Welch power spectral density estimation.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 5.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['psdwl']

_QUOTE = "Out of chaos, comes order. -- Friedrich Nietzsche"


def psdwl(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    nfft: int | None = None,
) -> DescriptiveResult:
    """Welch PSD estimation.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Segment length.
    noverlap : int or None
        Overlap samples (default nperseg//2).
    window : str
        Window function name.
    nfft : int or None
        FFT length (default nperseg).

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=float).ravel()
    seg = min(nperseg, len(x))
    f, psd = welch(x, fs=fs, nperseg=seg, noverlap=noverlap,
                   window=window, nfft=nfft)
    total_power = float(np.trapezoid(psd, f))

    return DescriptiveResult(
        name="psdwl",
        value=total_power,
        extra={"frequencies": f, "psd": psd, "total_power": total_power},
    )


def cheatsheet() -> str:
    return "psdwl({}) -> Welch PSD estimation."
