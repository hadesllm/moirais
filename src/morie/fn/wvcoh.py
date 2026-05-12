"""Wavelet coherence between two signals."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ._containers import DescriptiveResult

_QUOTE = "We are the spark that will light the fire."


def _morlet(t, f0=6.0):
    return np.exp(1j * 2 * np.pi * f0 * t) * np.exp(-(t**2) / 2) * np.pi ** (-0.25)


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    scales: np.ndarray | None = None,
) -> DescriptiveResult:
    r"""Wavelet coherence between two signals.

    .. math::

        C_{xy}(a, b) = \\frac{|S(W_x W_y^*)|^2}{S(|W_x|^2) \\cdot S(|W_y|^2)}

    where *S* is a smoothing operator.

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
        ``extra`` contains ``coherence``, ``scales``, ``frequencies``.
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
    smooth_len = max(3, int(scales.min()))
    kernel = np.ones(smooth_len) / smooth_len
    cross = Wx * np.conj(Wy)
    S_cross = np.zeros_like(cross)
    S_xx = np.zeros((len(scales), N))
    S_yy = np.zeros((len(scales), N))
    for i in range(len(scales)):
        S_cross[i] = fftconvolve(cross[i], kernel, mode="same")
        S_xx[i] = fftconvolve(np.abs(Wx[i]) ** 2, kernel, mode="same")
        S_yy[i] = fftconvolve(np.abs(Wy[i]) ** 2, kernel, mode="same")
    denom = S_xx * S_yy
    denom[denom < 1e-15] = 1e-15
    coherence = np.abs(S_cross) ** 2 / denom
    return DescriptiveResult(
        name="wavelet_coherence",
        value=float(np.mean(coherence)),
        extra={"coherence": coherence, "scales": scales, "frequencies": freqs},
    )


wvcoh = wavelet_coherence


def cheatsheet() -> str:
    return "_morlet({}) -> Wavelet coherence between two signals."
