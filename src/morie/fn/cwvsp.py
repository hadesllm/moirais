# morie.fn -- function file (hadesllm/morie)
"""CWT scalogram/spectrum using Morlet wavelet."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def _morlet_wavelet(t, omega0=6.0):
    return np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-(t**2) / 2)


def cwt_spectrum(x, scales=None, wavelet: str = "morlet") -> DescriptiveResult:
    """CWT scalogram/spectrum.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    scales : array-like or None
        Scales for CWT. Auto if None.
    wavelet : str
        Wavelet type (only 'morlet' supported).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if scales is None:
        scales = np.arange(1, min(N // 2, 128) + 1, dtype=float)
    else:
        scales = np.asarray(scales, dtype=float)

    cwt_matrix = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        half = int(4 * s)
        t_w = np.arange(-half, half + 1) / s
        if wavelet == "morlet":
            psi = _morlet_wavelet(t_w) / np.sqrt(s)
        else:
            raise ValueError(f"Unsupported wavelet '{wavelet}'")
        c = np.convolve(x, np.conj(psi), mode="same")
        cwt_matrix[i, :] = c[:N]

    power = np.abs(cwt_matrix) ** 2
    return DescriptiveResult(
        name="cwt_spectrum",
        value=float(np.max(power)),
        extra={"power": power, "coefficients": cwt_matrix, "scales": scales, "wavelet": wavelet},
    )


cwvsp = cwt_spectrum


def cheatsheet() -> str:
    return "_morlet_wavelet({}) -> CWT scalogram/spectrum using Morlet wavelet."
