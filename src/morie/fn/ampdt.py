# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Peak amplitude measurement at detected events."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It's a trap!"


def amplitude_detect(signal, peaks, **kwargs) -> DescriptiveResult:
    """Measure signal amplitude at detected peak locations.

    Parameters
    ----------
    signal : array-like
        Input signal.
    peaks : array-like of int
        Peak sample indices.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    valid = peaks[(peaks >= 0) & (peaks < len(signal))]
    amplitudes = signal[valid] if len(valid) > 0 else np.array([])
    mean_amp = float(np.mean(amplitudes)) if len(amplitudes) > 0 else 0.0
    return DescriptiveResult(
        name="amplitude_detect",
        value=mean_amp,
        extra={
            "amplitudes": amplitudes,
            "peak_indices": valid,
            "max_amplitude": float(np.max(amplitudes)) if len(amplitudes) > 0 else 0.0,
            "min_amplitude": float(np.min(amplitudes)) if len(amplitudes) > 0 else 0.0,
            "n_peaks": len(valid),
        },
    )


ampdt = amplitude_detect


def cheatsheet() -> str:
    return "amplitude_detect({}) -> Peak amplitude measurement at detected events."
