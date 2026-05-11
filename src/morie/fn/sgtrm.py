"""Signal trimming (segment extraction)."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Stay on target."


def trim_signal(x, start: int = 0, end: int | None = None) -> SignalResult:
    """Extract a segment from signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    start : int
        Start index. Default 0.
    end : int or None
        End index (exclusive). Default None (to end).

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    y = x[start:end]
    return SignalResult(
        name="trim_signal",
        filtered=y,
        fs=0.0,
        n_samples=len(y),
        extra={"start": start, "end": end},
    )


sgtrm = trim_signal


def cheatsheet() -> str:
    return "trim_signal({}) -> Signal trimming (segment extraction)."
