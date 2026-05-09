# moirais.fn — function file (hadesllm/moirais)
"""Kernel density estimate of event times."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "He who is brave is free. — Seneca"


def event_density(events, duration=1.0, bandwidth=0.1, **kwargs) -> DescriptiveResult:
    """Kernel density estimate of event times using Gaussian kernel.

    Parameters
    ----------
    events : array-like
        Event times.
    duration : float
        Total observation duration. Default 1.0.
    bandwidth : float
        Gaussian kernel bandwidth. Default 0.1.

    Returns
    -------
    DescriptiveResult
    """
    events = np.asarray(events, dtype=float)
    n_events = len(events)

    n_points = max(int(duration / (bandwidth / 5)), 100)
    t = np.linspace(0, duration, n_points)

    density = np.zeros(n_points)
    if n_events > 0:
        for ev in events:
            density += np.exp(-0.5 * ((t - ev) / bandwidth) ** 2)
        density /= n_events * bandwidth * np.sqrt(2 * np.pi)

    peak_density = float(np.max(density)) if n_events > 0 else 0.0

    return DescriptiveResult(
        name="event_density",
        value=peak_density,
        extra={
            "t": t,
            "density": density,
            "peak_density": peak_density,
            "n_events": n_events,
            "duration": duration,
            "bandwidth": bandwidth,
        },
    )


evtdn = event_density


def cheatsheet() -> str:
    return "event_density({}) -> Kernel density estimate of event times."
