# morie.fn -- function file (hadesllm/morie)
"""Pan-Tompkins QRS detection -- Rangayyan Ch 6."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_qrs_detect"]


def rangayyan_qrs_detect(x, fs=360.0):
    """Pan-Tompkins QRS detector.

    Pipeline (Pan & Tompkins 1985, adapted in Rangayyan Ch 6):

    1.  Bandpass 5–15 Hz (Butterworth IIR, zero-phase).
    2.  Differentiate ``y[n] = (1/8)(2 x[n] + x[n-1] - x[n-3] - 2 x[n-4])``.
    3.  Square.
    4.  Moving-window integration (window ≈ 150 ms).
    5.  Adaptive threshold = 0.3 × max(integrated) with refractory
        ≈ 200 ms; peaks above threshold are R-peaks.

    Parameters
    ----------
    x : array-like
        Raw ECG.
    fs : float
        Sampling rate (Hz, default 360 -- MIT-BIH).

    Returns
    -------
    RichResult with keys ``r_peaks`` (sample indices), ``rr_intervals_ms``,
    ``heart_rate_bpm``, ``integrated``, ``fs``.

    References
    ----------
    Pan & Tompkins (1985), IEEE TBME 32:230.  Rangayyan Ch 6.
    """
    from scipy.signal import butter, sosfiltfilt, find_peaks

    x = np.asarray(x, dtype=float).ravel()
    nyq = 0.5 * fs
    # 1) bandpass 5–15 Hz
    sos = butter(3, [5.0 / nyq, min(15.0, nyq * 0.95) / nyq], btype="band",
                 output="sos")
    bp = sosfiltfilt(sos, x)
    # 2) differentiate (Pan-Tompkins coefficients)
    der = np.zeros_like(bp)
    for n in range(4, bp.size):
        der[n] = (1.0 / 8.0) * (2 * bp[n] + bp[n - 1] - bp[n - 3] - 2 * bp[n - 4])
    # 3) square
    sq = der ** 2
    # 4) moving-window integration over 150 ms
    W = max(1, int(round(0.150 * fs)))
    kernel = np.ones(W) / W
    integ = np.convolve(sq, kernel, mode="same")
    # 5) detect peaks
    refractory = int(round(0.200 * fs))
    thr = 0.30 * integ.max() if integ.max() > 0 else 0.0
    peaks, _ = find_peaks(integ, height=thr, distance=max(1, refractory))
    # Optional: refine each peak to local max of bandpass-filtered signal
    half = int(round(0.05 * fs))
    refined = []
    for p in peaks:
        lo = max(0, p - half)
        hi = min(x.size, p + half + 1)
        refined.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    r_peaks = np.asarray(refined, dtype=int)
    rr_ms = np.diff(r_peaks) * (1000.0 / fs) if r_peaks.size > 1 else np.array([])
    hr = float(60000.0 / rr_ms.mean()) if rr_ms.size else float("nan")
    res = RichResult(
        title="QRS detection (Pan-Tompkins)",
        summary_lines=[
            ("Fs (Hz)", float(fs)),
            ("R-peaks", int(r_peaks.size)),
            ("Mean HR (bpm)", hr),
            ("Threshold", float(thr)),
        ],
        interpretation=(
            f"Detected {r_peaks.size} R-peaks; mean HR {hr:.1f} bpm."
        ),
        payload={
            "r_peaks": r_peaks,
            "rr_intervals_ms": rr_ms,
            "heart_rate_bpm": hr,
            "integrated": integ,
            "fs": float(fs),
        },
    )
    return with_describe_pointer(res, "rgqrs")


# CANONICAL TEST
# >>> fs = 360.0
# >>> t = np.arange(int(5*fs))/fs
# >>> # Simulated ECG-like impulse train at 1 Hz with Gaussian spikes
# >>> sig = np.zeros_like(t)
# >>> for tk in np.arange(0.5, 5.0, 1.0):
# ...     sig += np.exp(-((t - tk)*30)**2)
# >>> r = rangayyan_qrs_detect(sig, fs=fs)
# >>> 3 <= r["r_peaks"].size <= 6
# True


def cheatsheet():
    return "rgqrs: Pan-Tompkins QRS detector -- Rangayyan Ch 6"
