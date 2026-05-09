# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Adaptive segmentation via variance-based change detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def adaptive_segment(x, min_len: int = 50, threshold: float | None = None, **kwargs) -> DescriptiveResult:
    """Segment signal adaptively based on local variance changes.

    Parameters
    ----------
    x : array-like
        Input signal.
    min_len : int
        Minimum segment length.
    threshold : float or None
        Variance-ratio threshold for split. If None, uses 2.0.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if threshold is None:
        threshold = 2.0
    boundaries = [0]
    i = 0
    while i + 2 * min_len <= len(x):
        seg1 = x[i : i + min_len]
        seg2 = x[i + min_len : i + 2 * min_len]
        v1 = np.var(seg1)
        v2 = np.var(seg2)
        denom = max(v1, 1e-12)
        ratio = v2 / denom
        if ratio > threshold or (ratio < 1.0 / threshold and ratio > 0):
            boundaries.append(i + min_len)
            i += min_len
        else:
            i += 1
    boundaries.append(len(x))
    boundaries = sorted(set(boundaries))
    n_segments = len(boundaries) - 1
    return DescriptiveResult(
        name="adaptive_segment",
        value=float(n_segments),
        extra={
            "boundaries": np.array(boundaries),
            "n_segments": n_segments,
            "min_len": min_len,
            "threshold": threshold,
        },
    )


adseg = adaptive_segment


def cheatsheet() -> str:
    return "adaptive_segment({}) -> Adaptive segmentation via variance-based change detection."
