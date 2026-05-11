# morie.fn — function file (hadesllm/morie)
"""Filter events by duration."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "By all means, marry. If you get a good wife, you'll become happy; if you get a bad one, you'll become a philosopher. — Socrates"


def duration_detect(events, min_dur=0.01, max_dur=1.0, **kwargs) -> DescriptiveResult:
    """Filter events by duration.

    Parameters
    ----------
    events : list of tuples (start, end)
        Event boundaries as ``(start_time, end_time)`` pairs.
    min_dur : float
        Minimum event duration. Default 0.01.
    max_dur : float
        Maximum event duration. Default 1.0.

    Returns
    -------
    DescriptiveResult
    """
    events = [(float(s), float(e)) for s, e in events]
    durations = np.array([e - s for s, e in events])

    mask = (durations >= min_dur) & (durations <= max_dur)
    kept = [ev for ev, m in zip(events, mask) if m]
    kept_dur = durations[mask]

    return DescriptiveResult(
        name="duration_detect",
        value=float(len(kept)),
        extra={
            "kept_events": kept,
            "kept_durations": kept_dur,
            "rejected": int(np.sum(~mask)),
            "min_dur": min_dur,
            "max_dur": max_dur,
        },
    )


durdt = duration_detect


def cheatsheet() -> str:
    return "duration_detect({}) -> Filter events by duration."
