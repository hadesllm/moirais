# morie.fn — function file (hadesllm/morie)
"""Pan-Tompkins QRS detector."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def pan_tompkins(
    ecg: np.ndarray,
    fs: float,
) -> SignalResult:
    """Pan-Tompkins QRS detection algorithm.

    Returns R-peak sample indices in ``extra["r_peaks"]``.

    :param ecg: 1-D ECG signal.
    :param fs: Sampling frequency in Hz.
    :return: SignalResult with R-peak indices.
    """
    from scipy.signal import butter, sosfiltfilt

    ecg = np.asarray(ecg, dtype=float).ravel()
    n = len(ecg)

    sos = butter(2, [5, 15], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, ecg)

    diff = np.diff(filtered)
    diff = np.append(diff, 0)

    squared = diff**2

    win_len = int(0.15 * fs)
    if win_len < 1:
        win_len = 1
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode="same")

    threshold = 0.5 * np.max(integrated)
    candidates = np.where(integrated > threshold)[0]

    if len(candidates) == 0:
        return SignalResult(
            name="pan_tompkins",
            filtered=ecg,
            fs=fs,
            n_samples=n,
            extra={"r_peaks": np.array([], dtype=int)},
        )

    min_dist = int(0.3 * fs)
    r_peaks = [candidates[0]]
    for c in candidates[1:]:
        if c - r_peaks[-1] >= min_dist:
            r_peaks.append(c)

    r_peaks_refined = []
    search_win = int(0.075 * fs)
    for pk in r_peaks:
        lo = max(0, pk - search_win)
        hi = min(n, pk + search_win + 1)
        r_peaks_refined.append(lo + int(np.argmax(ecg[lo:hi])))

    r_peaks_arr = np.array(r_peaks_refined, dtype=int)
    return SignalResult(
        name="pan_tompkins",
        filtered=ecg,
        fs=fs,
        n_samples=n,
        extra={"r_peaks": r_peaks_arr, "n_peaks": len(r_peaks_arr)},
    )


ecgdet = pan_tompkins


def cheatsheet() -> str:
    return "pan_tompkins({}) -> Pan-Tompkins QRS detector."
