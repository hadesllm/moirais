# morie.fn — function file (hadesllm/morie)
"""Pan-Tompkins QRS detection."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def pan_tompkins_qrs(ecg, fs: float = 360.0) -> SignalResult:
    """Detect QRS complexes using Pan-Tompkins algorithm.

    Parameters
    ----------
    ecg : array-like
        ECG signal.
    fs : float
        Sampling frequency in Hz. Default 360.

    Returns
    -------
    SignalResult
    """
    from morie._detection import pan_tompkins_qrs as _pt

    ecg = np.asarray(ecg, dtype=float)
    qrs = _pt(ecg, fs=fs)
    hr = len(qrs) * fs / len(ecg) * 60 if len(ecg) > 0 else 0.0
    return SignalResult(
        name="pan_tompkins_qrs",
        filtered=ecg,
        fs=fs,
        n_samples=len(ecg),
        extra={"qrs_indices": qrs, "num_beats": len(qrs), "heart_rate": hr},
    )


pntmp = pan_tompkins_qrs


def cheatsheet() -> str:
    return "pan_tompkins_qrs({}) -> Pan-Tompkins QRS detection."
