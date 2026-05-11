# morie.fn — function file (hadesllm/morie)
"""QRS complex detection (Pan-Tompkins algorithm).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 11.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['qrsdt']
def qrsdt(
    x: np.ndarray,
    fs: float = 360.0,
    *,
    low: float = 5.0,
    high: float = 15.0,
    integration_window: float = 0.15,
    refractory: float = 0.2,
) -> DescriptiveResult:
    """Detect QRS complexes using Pan-Tompkins.

    Parameters
    ----------
    x : array-like
        1-D ECG signal.
    fs : float
        Sampling frequency in Hz.
    low : float
        Bandpass lower cutoff (Hz).
    high : float
        Bandpass upper cutoff (Hz).
    integration_window : float
        Moving-average window duration (seconds).
    refractory : float
        Refractory period (seconds).

    Returns
    -------
    DescriptiveResult
        ``extra`` has ``r_peaks``, ``rr_intervals``, ``heart_rate_bpm``.
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    nyq = fs / 2.0

    lo = max(low / nyq, 1e-6)
    hi = min(high / nyq, 1.0 - 1e-6)
    sos = butter(2, [lo, hi], btype="band", output="sos")
    filtered = sosfiltfilt(sos, x)

    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2

    win = max(1, int(integration_window * fs))
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")

    threshold = 0.5 * np.max(integrated)
    refr_samples = int(refractory * fs)

    r_peaks = []
    i = 0
    while i < n:
        if integrated[i] > threshold:
            region_start = i
            while i < n and integrated[i] > threshold:
                i += 1
            region = integrated[region_start:i]
            peak = region_start + np.argmax(region)
            if len(r_peaks) == 0 or (peak - r_peaks[-1]) >= refr_samples:
                r_peaks.append(peak)
        else:
            i += 1

    r_peaks = np.array(r_peaks, dtype=int)
    rr = np.diff(r_peaks) / fs if len(r_peaks) > 1 else np.array([])
    hr = 60.0 / rr if len(rr) > 0 else np.array([])

    return DescriptiveResult(
        name="qrsdt",
        value=float(len(r_peaks)),
        extra={
            "r_peaks": r_peaks,
            "rr_intervals": rr,
            "heart_rate_bpm": hr,
            "mean_hr": float(np.mean(hr)) if len(hr) > 0 else 0.0,
        },
    )


def cheatsheet() -> str:
    return "qrsdt({}) -> QRS detection (Pan-Tompkins)."
