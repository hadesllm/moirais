# morie.fn — function file (hadesllm/morie)
"""Reassigned spectrogram for sharper TF representation."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft as _stft

from ._containers import DescriptiveResult

_QUOTE = "Let the past die. Kill it, if you have to."


def reassigned_stft(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> DescriptiveResult:
    """Reassigned Short-Time Fourier Transform spectrogram.

    Sharpens time-frequency localization by reassigning each spectrogram
    coefficient to its center of gravity in the TF plane.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).
    nperseg : int
        Segment length (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``spectrogram``, ``frequencies``, ``times``,
        ``reassigned_times``, ``reassigned_freqs``.
    """
    x = np.asarray(x, dtype=float).ravel()
    nperseg = min(nperseg, len(x))
    f, t, Zxx = _stft(x, fs=fs, nperseg=nperseg, return_onesided=True)
    power = np.abs(Zxx) ** 2
    window = np.hanning(nperseg)
    t_window = np.arange(nperseg) / fs
    dh = window * t_window
    f_dh, t_dh, Zxx_dh = _stft(
        x, fs=fs, window=dh / (np.sum(dh**2) ** 0.5 + 1e-12), nperseg=nperseg, return_onesided=True
    )
    mask = np.abs(Zxx) > 1e-10 * np.max(np.abs(Zxx))
    r_times = np.zeros_like(power)
    r_freqs = np.zeros_like(power)
    for i in range(len(t)):
        r_times[:, i] = t[i]
        r_freqs[:, i] = f
    denom = Zxx.copy()
    denom[~mask] = 1.0
    correction = Zxx_dh / denom
    correction[~mask] = 0.0
    r_times = r_times + np.real(correction)
    return DescriptiveResult(
        name="reassigned_stft",
        value=float(len(f)),
        extra={
            "spectrogram": power,
            "frequencies": f,
            "times": t,
            "reassigned_times": r_times,
            "reassigned_freqs": r_freqs,
        },
    )


reasn = reassigned_stft


def cheatsheet() -> str:
    return "reassigned_stft({}) -> Reassigned spectrogram for sharper TF representation."
