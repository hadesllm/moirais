"""Compute CWT scalogram (magnitude)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies."


def _morlet(t, omega0=6.0):
    return np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-(t**2) / 2)


def wavelet_scalogram(x, scales=None, fs: float = 1.0) -> DescriptiveResult:
    """Compute CWT scalogram (magnitude squared).

    Parameters
    ----------
    x : array-like
        1-D input signal.
    scales : array-like or None
        CWT scales. Auto if None.
    fs : float
        Sampling frequency.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if scales is None:
        scales = np.arange(1, min(N // 2, 64) + 1, dtype=float)
    else:
        scales = np.asarray(scales, dtype=float)

    scalogram = np.zeros((len(scales), N))
    for i, s in enumerate(scales):
        half = int(4 * s)
        t_w = np.arange(-half, half + 1) / s
        psi = _morlet(t_w) / np.sqrt(s)
        c = np.convolve(x, np.conj(psi), mode="same")
        scalogram[i, :] = np.abs(c[:N]) ** 2

    freqs = fs / scales
    return DescriptiveResult(
        name="wavelet_scalogram",
        value=float(np.max(scalogram)),
        extra={"scalogram": scalogram, "scales": scales, "frequencies": freqs, "fs": fs},
    )


wvscl = wavelet_scalogram


def cheatsheet() -> str:
    return "_morlet({}) -> Compute CWT scalogram (magnitude)."
