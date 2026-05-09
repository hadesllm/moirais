# moirais.fn — function file (hadesllm/moirais)
"""Event alignment via cross-correlation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Real knowledge is to know the extent of one's ignorance. — Confucius"


def event_align(signal, events, window: int = 100, **kwargs) -> DescriptiveResult:
    """Align detected events to template by cross-correlation.

    Parameters
    ----------
    signal : array-like
        Input signal.
    events : array-like of int
        Event indices in the signal.
    window : int
        Half-window size around each event.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    events = np.asarray(events, dtype=int)
    if len(events) == 0:
        return DescriptiveResult(
            name="event_align",
            value=0.0,
            extra={"aligned_events": np.array([], dtype=int), "shifts": np.array([])},
        )
    segments = []
    for ev in events:
        lo = max(0, ev - window)
        hi = min(len(signal), ev + window)
        segments.append(signal[lo:hi])
    max_len = max(len(s) for s in segments)
    padded = np.zeros((len(segments), max_len))
    for i, s in enumerate(segments):
        padded[i, : len(s)] = s
    template = np.mean(padded, axis=0)
    shifts = []
    aligned = []
    for i, ev in enumerate(events):
        lo = max(0, ev - window)
        hi = min(len(signal), ev + window)
        seg = signal[lo:hi]
        corr = np.correlate(seg, template[: len(seg)], mode="full")
        shift = int(np.argmax(corr)) - (len(seg) - 1)
        shifts.append(shift)
        aligned.append(ev + shift)
    return DescriptiveResult(
        name="event_align",
        value=float(np.mean(np.abs(shifts))),
        extra={
            "aligned_events": np.array(aligned),
            "shifts": np.array(shifts),
            "template": template,
        },
    )


evtag = event_align


def cheatsheet() -> str:
    return "event_align({}) -> Event alignment via cross-correlation."
