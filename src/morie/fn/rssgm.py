# morie.fn — function file (hadesllm/morie)
"""Reassigned spectrogram (time-frequency reassignment)."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from ._containers import DescriptiveResult

_QUOTE = "Knowledge itself is power. — Francis Bacon"


def reassigned_spectrogram(
    x,
    fs: float = 1.0,
    window: int = 256,
    hop: int = 128,
) -> DescriptiveResult:
    """Compute reassigned spectrogram for sharper time-frequency localisation.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency.
    window : int
        Window length in samples.
    hop : int
        Hop size in samples.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    f, t, Zxx = stft(x, fs=fs, nperseg=window, noverlap=window - hop)
    magnitude = np.abs(Zxx)

    phase = np.angle(Zxx)
    dt_phase = np.zeros_like(phase)
    if phase.shape[1] > 1:
        dt_phase[:, 1:] = np.diff(phase, axis=1)
    df_phase = np.zeros_like(phase)
    if phase.shape[0] > 1:
        df_phase[1:, :] = np.diff(phase, axis=0)

    t_reassigned = np.zeros_like(magnitude)
    for j in range(len(t)):
        t_reassigned[:, j] = t[j]
    if phase.shape[1] > 1:
        t_step = t[1] - t[0] if len(t) > 1 else 1.0
        t_reassigned -= dt_phase / (2 * np.pi) * t_step

    f_reassigned = np.zeros_like(magnitude)
    for i in range(len(f)):
        f_reassigned[i, :] = f[i]
    if phase.shape[0] > 1:
        f_step = f[1] - f[0] if len(f) > 1 else 1.0
        f_reassigned += df_phase / (2 * np.pi) * f_step

    return DescriptiveResult(
        name="reassigned_spectrogram",
        value=float(np.max(magnitude)),
        extra={
            "magnitude": magnitude,
            "frequencies": f,
            "times": t,
            "t_reassigned": t_reassigned,
            "f_reassigned": f_reassigned,
        },
    )


rssgm = reassigned_spectrogram


def cheatsheet() -> str:
    return "reassigned_spectrogram({}) -> Reassigned spectrogram (time-frequency reassignment)."
