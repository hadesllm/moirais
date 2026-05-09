"""Split signal into non-overlapping segments."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is where the fun begins."


def split_signal(x, segment_len: int) -> DescriptiveResult:
    """Split signal into non-overlapping segments.

    Parameters
    ----------
    x : array-like
        Input signal.
    segment_len : int
        Length of each segment.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n_full = len(x) // segment_len
    segments = x[: n_full * segment_len].reshape(n_full, segment_len)
    remainder = x[n_full * segment_len :]
    return DescriptiveResult(
        name="split_signal",
        value=float(n_full),
        extra={"segments": segments, "remainder": remainder, "segment_len": segment_len},
    )


sgspl = split_signal


def cheatsheet() -> str:
    return "split_signal({}) -> Split signal into non-overlapping segments."
