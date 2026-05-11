# morie.fn — function file (hadesllm/morie)
"""Event latency measurement from stimulus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You must unlearn what you have learned."


def latency_detect(signal, events, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Measure latency from stimulus onset to peak response.

    Parameters
    ----------
    signal : array-like
        Input signal (e.g., evoked potential).
    events : array-like of int
        Stimulus onset sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    events = np.asarray(events, dtype=int)
    latencies = []
    for ev in events:
        if ev < 0 or ev >= len(signal):
            continue
        seg = signal[ev:]
        if len(seg) < 2:
            continue
        peak_idx = int(np.argmax(np.abs(seg)))
        latencies.append(peak_idx / fs)
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    return DescriptiveResult(
        name="latency_detect",
        value=mean_lat,
        extra={
            "latencies": latencies,
            "mean_latency": mean_lat,
            "std_latency": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
            "n_events": len(latencies),
            "fs": fs,
        },
    )


latdt = latency_detect


def cheatsheet() -> str:
    return "latency_detect({}) -> Event latency measurement from stimulus."
