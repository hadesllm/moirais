# morie.fn — function file (hadesllm/morie)
"""Threshold-based peak detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


def peak_detect_threshold(x, threshold: float | None = None, min_dist: int = 1, **kwargs) -> DescriptiveResult:
    """Detect peaks exceeding a threshold with minimum distance constraint.

    Parameters
    ----------
    x : array-like
        Input signal.
    threshold : float or None
        Amplitude threshold. If None, uses mean + 1 std.
    min_dist : int
        Minimum distance between peaks in samples.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if threshold is None:
        threshold = float(np.mean(x) + np.std(x))
    candidates = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            candidates.append(i)
    peaks = []
    for c in candidates:
        if len(peaks) == 0 or (c - peaks[-1]) >= min_dist:
            peaks.append(c)
    peaks = np.array(peaks, dtype=int)
    amplitudes = x[peaks] if len(peaks) > 0 else np.array([])
    return DescriptiveResult(
        name="peak_detect_threshold",
        value=float(len(peaks)),
        extra={
            "peaks": peaks,
            "amplitudes": amplitudes,
            "threshold": threshold,
            "min_dist": min_dist,
        },
    )


pdthr = peak_detect_threshold


def cheatsheet() -> str:
    return "peak_detect_threshold({}) -> Threshold-based peak detection."
