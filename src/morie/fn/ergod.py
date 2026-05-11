# morie.fn — function file (hadesllm/morie)
"""Ergodicity test."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def ergodicity_test(x, n_segments=10, **kwargs) -> DescriptiveResult:
    """Test ergodicity via time vs ensemble averages.

    Splits the signal into segments, computes per-segment means,
    and checks consistency with the overall mean.

    Parameters
    ----------
    x : array-like
        Input signal.
    n_segments : int
        Number of segments to split into.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if len(x) < n_segments:
        raise ValueError("Signal too short for requested segments.")
    seg_len = len(x) // n_segments
    segments = [x[i * seg_len : (i + 1) * seg_len] for i in range(n_segments)]
    seg_means = np.array([np.mean(s) for s in segments])
    overall_mean = float(np.mean(x))
    seg_var = float(np.var(seg_means))
    ergodic = bool(seg_var < np.var(x) / n_segments * 4)
    return DescriptiveResult(
        name="ergodicity_test",
        value=seg_var,
        extra={
            "overall_mean": overall_mean,
            "segment_means": seg_means.tolist(),
            "segment_variance": seg_var,
            "ergodic": ergodic,
            "n_segments": n_segments,
        },
    )


ergod = ergodicity_test


def cheatsheet() -> str:
    return "ergodicity_test({}) -> Ergodicity test."
