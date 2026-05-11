# morie.fn — function file (hadesllm/morie)
"""Continuous wavelet scalogram.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 8.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['cwtsc']

_QUOTE = "Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu"


def cwtsc(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    wavelet: str = "morlet",
    n_scales: int = 64,
    f_min: float | None = None,
    f_max: float | None = None,
) -> DescriptiveResult:
    """Continuous wavelet transform scalogram.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    wavelet : str
        Wavelet name (``'morlet'`` supported).
    n_scales : int
        Number of scales.
    f_min : float or None
        Minimum frequency (default 1 Hz).
    f_max : float or None
        Maximum frequency (default fs/2).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if f_min is None:
        f_min = 1.0
    if f_max is None:
        f_max = fs / 2.0

    omega0 = 6.0
    freqs = np.linspace(f_min, f_max, n_scales)
    scales = omega0 * fs / (2 * np.pi * freqs)

    t = np.arange(n) / fs
    coeffs = np.zeros((n_scales, n), dtype=complex)

    for i, s in enumerate(scales):
        half = int(min(5 * s * fs, n // 2))
        tau = np.arange(-half, half + 1) / fs
        psi = (np.pi ** -0.25) * np.exp(1j * omega0 * tau / s) * np.exp(-tau ** 2 / (2 * s ** 2))
        psi /= np.sqrt(s)
        coeffs[i] = np.convolve(x, psi, mode="same")[:n]

    scalogram = np.abs(coeffs) ** 2

    return DescriptiveResult(
        name="cwtsc",
        value=float(np.mean(scalogram)),
        extra={
            "scalogram": scalogram,
            "frequencies": freqs,
            "scales": scales,
            "times": t,
        },
    )


def cheatsheet() -> str:
    return "cwtsc({}) -> CWT scalogram."
