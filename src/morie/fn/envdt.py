# morie.fn — function file (hadesllm/morie)
"""Signal envelope via rectification and lowpass."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this."


def envelope_detect(x, cutoff_ratio: float = 0.05, **kwargs) -> DescriptiveResult:
    """Compute signal envelope via full-wave rectification and lowpass filter.

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff_ratio : float
        Lowpass cutoff as fraction of signal length for moving average.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    rectified = np.abs(x)
    win_len = max(1, int(len(x) * cutoff_ratio))
    kernel = np.ones(win_len) / win_len
    envelope = np.convolve(rectified, kernel, mode="same")
    return DescriptiveResult(
        name="envelope_detect",
        value=float(np.max(envelope)),
        extra={
            "envelope": envelope,
            "rectified": rectified,
            "window_length": win_len,
        },
    )


envdt = envelope_detect


def cheatsheet() -> str:
    return "envelope_detect({}) -> Signal envelope via rectification and lowpass."
