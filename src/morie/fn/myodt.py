# morie.fn -- function file (hadesllm/morie)
"""EMG onset detection via double threshold."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def myogram_onset(
    x, fs: float = 1000.0, threshold_factor: float = 3.0, window_ms: float = 50.0, **kwargs
) -> DescriptiveResult:
    """Detect EMG muscle activation onset using double threshold method.

    Parameters
    ----------
    x : array-like
        EMG signal.
    fs : float
        Sampling frequency in Hz.
    threshold_factor : float
        Multiplier of baseline RMS for threshold.
    window_ms : float
        Window length in milliseconds for RMS calculation.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    win = max(1, int(fs * window_ms / 1000.0))
    rms = np.zeros(len(x))
    for i in range(len(x)):
        lo = max(0, i - win // 2)
        hi = min(len(x), i + win // 2 + 1)
        rms[i] = np.sqrt(np.mean(x[lo:hi] ** 2))
    baseline_len = min(len(rms), max(1, int(0.1 * len(rms))))
    baseline_rms = np.mean(rms[:baseline_len])
    thresh = baseline_rms * threshold_factor
    onsets = []
    active = False
    for i in range(len(rms)):
        if not active and rms[i] > thresh:
            onsets.append(i)
            active = True
        elif active and rms[i] < thresh * 0.5:
            active = False
    return DescriptiveResult(
        name="myogram_onset",
        value=float(len(onsets)),
        extra={
            "onsets": np.array(onsets, dtype=int),
            "rms": rms,
            "threshold": thresh,
            "fs": fs,
        },
    )


myodt = myogram_onset


def cheatsheet() -> str:
    return "myogram_onset({}) -> EMG onset detection via double threshold."
