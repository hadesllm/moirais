"""Synchrosqueezed transform of CWT."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficulties strengthen the mind, as labor does the body. — Seneca"


def _morlet(t, omega0=6.0):
    return np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-(t**2) / 2)


def synchrosqueezed_transform(x, fs: float = 1.0) -> DescriptiveResult:
    """Synchrosqueezing of CWT coefficients.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    scales = np.arange(2, min(N // 2, 64) + 1, dtype=float)
    n_scales = len(scales)
    cwt_matrix = np.zeros((n_scales, N), dtype=complex)
    for i, s in enumerate(scales):
        t_wavelet = np.arange(-int(4 * s), int(4 * s) + 1) / s
        wavelet = _morlet(t_wavelet) / np.sqrt(s)
        conv = np.convolve(x, wavelet, mode="same")
        cwt_matrix[i, :] = conv[:N]

    phase = np.angle(cwt_matrix)
    inst_freq = np.zeros_like(phase)
    inst_freq[:, 1:] = np.diff(phase, axis=1) * fs / (2 * np.pi)

    freqs = fs / scales
    n_freq_bins = n_scales
    sst = np.zeros((n_freq_bins, N), dtype=complex)
    for i in range(n_scales):
        for j in range(N):
            if np.abs(cwt_matrix[i, j]) < 1e-10:
                continue
            w = inst_freq[i, j]
            idx = np.argmin(np.abs(freqs - abs(w)))
            sst[idx, j] += cwt_matrix[i, j]

    return DescriptiveResult(
        name="synchrosqueezed_transform",
        value=float(np.max(np.abs(sst))),
        extra={"sst": np.abs(sst), "frequencies": freqs, "scales": scales, "cwt": np.abs(cwt_matrix)},
    )


sqsgm = synchrosqueezed_transform


def cheatsheet() -> str:
    return "_morlet({}) -> Synchrosqueezed transform of CWT."
