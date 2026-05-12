"""ST segment level analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Patience is bitter, but its fruit is sweet. -- Aristotle"


def st_segment(signal, qrs_off, t_on, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Analyze ST segment level between QRS offset and T-wave onset.

    Parameters
    ----------
    signal : array-like
        ECG signal.
    qrs_off : array-like of int
        QRS offset (J-point) sample indices.
    t_on : array-like of int
        T-wave onset sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    qrs_off = np.asarray(qrs_off, dtype=int)
    t_on = np.asarray(t_on, dtype=int)
    n = min(len(qrs_off), len(t_on))
    if n == 0:
        return DescriptiveResult(
            name="st_segment",
            value=0.0,
            extra={"st_levels": np.array([]), "st_slopes": np.array([])},
        )
    levels = []
    slopes = []
    for i in range(n):
        lo = int(qrs_off[i])
        hi = int(t_on[i])
        if lo >= hi or lo < 0 or hi > len(signal):
            levels.append(0.0)
            slopes.append(0.0)
            continue
        seg = signal[lo:hi]
        levels.append(float(np.mean(seg)))
        if len(seg) > 1:
            t = np.arange(len(seg)) / fs
            slope = (seg[-1] - seg[0]) / (t[-1] - t[0]) if t[-1] != t[0] else 0.0
            slopes.append(float(slope))
        else:
            slopes.append(0.0)
    return DescriptiveResult(
        name="st_segment",
        value=float(np.mean(levels)),
        extra={
            "st_levels": np.array(levels),
            "st_slopes": np.array(slopes),
            "n_segments": n,
            "fs": fs,
        },
    )


stsgm = st_segment


def cheatsheet() -> str:
    return "st_segment({}) -> ST segment level analysis."
