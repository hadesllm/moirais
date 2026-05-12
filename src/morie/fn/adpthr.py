# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Adaptive threshold based on local statistics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "No man ever steps in the same river twice. -- Heraclitus"


def adaptive_threshold_detect(x, window=50, k=1.5, **kwargs) -> DescriptiveResult:
    """Adaptive threshold detection based on local mean and std.

    A sample is flagged when it exceeds ``local_mean + k * local_std``.

    Parameters
    ----------
    x : array-like
        Input signal.
    window : int
        Rolling window size. Default 50.
    k : float
        Number of local standard deviations for threshold. Default 1.5.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    threshold = np.zeros(n)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        threshold[i] = np.mean(seg) + k * np.std(seg)

    detections = np.where(x > threshold)[0]

    return DescriptiveResult(
        name="adaptive_threshold_detect",
        value=float(len(detections)),
        extra={
            "detections": detections,
            "threshold": threshold,
            "window": window,
            "k": k,
        },
    )


adpthr = adaptive_threshold_detect


def cheatsheet() -> str:
    return "adaptive_threshold_detect({}) -> Adaptive threshold based on local statistics."
