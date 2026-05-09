"""Cross-wavelet spectrum."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ._containers import DescriptiveResult

_QUOTE = "Your eyes can deceive you. Don't trust them."


def _morlet(t, f0=6.0):
    return np.exp(1j * 2 * np.pi * f0 * t) * np.exp(-(t**2) / 2) * np.pi ** (-0.25)


def cross_wavelet(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    scales: np.ndarray | None = None,
) -> DescriptiveResult:
    """Cross-wavelet spectrum of two signals.

    .. math::

        W_{xy}(a, b) = W_x(a, b) \\cdot W_y^*(a, b)

    Parameters
    ----------
    x, y : array-like
        Input signals (same length).
    fs : float
        Sampling frequency (default 1.0).
    scales : array-like or None
        CWT scales. Auto-generated if None.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``cross_spectrum``, ``power``, ``phase``,
        ``scales``, ``frequencies``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    N = len(x)
    if scales is None:
        scales = np.arange(1, min(N // 2, 64) + 1, dtype=float)
    freqs = fs / (scales * (2 * np.pi / 6.0))
    Wx = np.zeros((len(scales), N), dtype=complex)
    Wy = np.zeros_like(Wx)
    for i, s in enumerate(scales):
        t = np.arange(-int(4 * s), int(4 * s) + 1) / fs
        psi = _morlet(t / s) / np.sqrt(s)
        Wx[i] = fftconvolve(x, psi[::-1], mode="same")
        Wy[i] = fftconvolve(y, psi[::-1], mode="same")
    cross = Wx * np.conj(Wy)
    power = np.abs(cross)
    phase = np.angle(cross)
    return DescriptiveResult(
        name="cross_wavelet",
        value=float(np.mean(power)),
        extra={
            "cross_spectrum": cross,
            "power": power,
            "phase": phase,
            "scales": scales,
            "frequencies": freqs,
        },
    )


xwvlt = cross_wavelet


def cheatsheet() -> str:
    return "_morlet({}) -> Cross-wavelet spectrum."
