"""Synchrosqueezed CWT."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ._containers import DescriptiveResult

_QUOTE = "You know, no matter how much we fought, I always hated watching you leave."


def synchrosqueeze(
    x: np.ndarray,
    fs: float = 1.0,
    scales: np.ndarray | None = None,
) -> DescriptiveResult:
    """Synchrosqueezed Continuous Wavelet Transform.

    Reassigns CWT coefficients to instantaneous frequency bins for
    improved frequency resolution.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).
    scales : array-like or None
        CWT scales. Auto-generated if None.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``ssq``, ``frequencies``, ``scales``.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if scales is None:
        scales = np.arange(1, min(N // 2, 64) + 1, dtype=float)
    f0 = 6.0
    Wx = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        t = np.arange(-int(4 * s), int(4 * s) + 1) / fs
        psi = np.exp(1j * 2 * np.pi * f0 * t / s) * np.exp(-(t**2) / (2 * s**2)) * np.pi ** (-0.25) / np.sqrt(s)
        Wx[i] = fftconvolve(x, psi[::-1], mode="same")
    dWx = np.zeros_like(Wx)
    for i in range(len(scales)):
        dWx[i] = np.gradient(Wx[i]) * fs
    mask = np.abs(Wx) > 1e-10 * np.max(np.abs(Wx))
    inst_freq = np.zeros((len(scales), N))
    inst_freq[mask] = np.imag(dWx[mask] / (2j * np.pi * Wx[mask]))
    freqs = fs * f0 / (2 * np.pi * scales)
    n_freq = len(freqs)
    ssq = np.zeros((n_freq, N))
    for i in range(len(scales)):
        for j in range(N):
            if mask[i, j]:
                fi = inst_freq[i, j]
                k = np.argmin(np.abs(freqs - fi))
                ssq[k, j] += np.abs(Wx[i, j]) ** 2
    return DescriptiveResult(
        name="synchrosqueeze",
        value=float(len(scales)),
        extra={"ssq": ssq, "frequencies": freqs, "scales": scales},
    )


sqztf = synchrosqueeze


def cheatsheet() -> str:
    return "synchrosqueeze({}) -> Synchrosqueezed CWT."
