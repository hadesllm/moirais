# moirais.fn — function file (hadesllm/moirais)
"""EMG RMS-based onset detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def emg_rms_threshold(x, window: int = 50, threshold_factor: float = 2.0, **kwargs) -> DescriptiveResult:
    """Detect EMG onset using sliding-window RMS and threshold.

    Parameters
    ----------
    x : array-like
        EMG signal.
    window : int
        Sliding window size in samples.
    threshold_factor : float
        Multiplier of baseline RMS for detection.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    rms = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        rms[i] = np.sqrt(np.mean(x[lo:hi] ** 2))
    baseline_n = max(1, min(window * 2, n // 5))
    baseline = np.mean(rms[:baseline_n])
    thresh = baseline * threshold_factor
    onsets = []
    offsets = []
    active = False
    for i in range(n):
        if not active and rms[i] > thresh:
            onsets.append(i)
            active = True
        elif active and rms[i] < thresh:
            offsets.append(i)
            active = False
    return DescriptiveResult(
        name="emg_rms_threshold",
        value=float(len(onsets)),
        extra={
            "onsets": np.array(onsets, dtype=int),
            "offsets": np.array(offsets, dtype=int),
            "rms": rms,
            "threshold": thresh,
            "window": window,
        },
    )


emgrt = emg_rms_threshold


def cheatsheet() -> str:
    return "emg_rms_threshold({}) -> EMG RMS-based onset detection."
