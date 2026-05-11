"""Frame signal into overlapping windows."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I cannot teach anybody anything. I can only make them think. — Socrates"


def frame_signal(x, frame_len: int, hop_len: int) -> DescriptiveResult:
    """Split signal into overlapping frames.

    Parameters
    ----------
    x : array-like
        Input signal.
    frame_len : int
        Length of each frame.
    hop_len : int
        Hop (stride) between frames.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n_frames = 1 + (len(x) - frame_len) // hop_len
    if n_frames < 1:
        raise ValueError("Signal too short for given frame_len and hop_len.")
    indices = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    frames = x[indices]
    return DescriptiveResult(
        name="frame_signal",
        value=float(n_frames),
        extra={"frames": frames, "frame_len": frame_len, "hop_len": hop_len},
    )


sgfrm = frame_signal


def cheatsheet() -> str:
    return "frame_signal({}) -> Frame signal into overlapping windows."
