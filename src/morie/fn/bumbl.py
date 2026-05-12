# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Without music, life would be a mistake. -- Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_centroid(
    x: np.ndarray,
    *,
    fs: float = 1.0,
    nfft: int | None = None,
) -> DescriptiveResult:
    """Compute the spectral centroid of a signal.

    The spectral centroid is the weighted mean frequency:
    SC = sum(f * |X(f)|) / sum(|X(f)|)

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    nfft : int, optional
        FFT length. Defaults to len(x).

    Returns
    -------
    DescriptiveResult
        With ``value`` = spectral centroid in Hz.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 2:
        raise ValueError("Signal must have at least 2 samples")
    if nfft is None:
        nfft = len(x)
    spectrum = np.abs(np.fft.rfft(x, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    total = spectrum.sum()
    if total == 0:
        centroid = 0.0
    else:
        centroid = float(np.sum(freqs * spectrum) / total)
    return DescriptiveResult(
        name="spectral_centroid",
        value=centroid,
        extra={"fs": fs, "nfft": nfft, "n": len(x)},
    )


bumbl = spectral_centroid


def cheatsheet() -> str:
    return "spectral_centroid({}) -> Spectral centroid (buzz frequency). 'I sting like a bee.' --"
