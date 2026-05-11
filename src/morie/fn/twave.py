"""T-wave detection in ECG signals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def t_wave_detect(ecg, qrs_locs, fs: float = 360.0) -> DescriptiveResult:
    """Detect T-wave peaks given QRS locations.

    Parameters
    ----------
    ecg : array-like
        ECG signal.
    qrs_locs : array-like
        QRS complex locations (sample indices).
    fs : float
        Sampling frequency in Hz. Default 360.

    Returns
    -------
    DescriptiveResult
    """
    from morie._detection import t_wave_detect as _tw

    ecg = np.asarray(ecg, dtype=float)
    qrs_locs = np.asarray(qrs_locs, dtype=int)
    peaks = _tw(ecg, qrs_locs, fs=fs)
    return DescriptiveResult(
        name="t_wave_detect",
        value=len(peaks),
        extra={"t_peak_indices": peaks},
    )


twave = t_wave_detect


def cheatsheet() -> str:
    return "t_wave_detect({}) -> T-wave detection in ECG signals."
