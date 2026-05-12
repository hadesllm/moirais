# morie.fn -- function file (hadesllm/morie)
"""Inter-spike interval analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def isi_analyze(spike_times, **kwargs) -> DescriptiveResult:
    """Inter-spike interval (ISI) analysis.

    Parameters
    ----------
    spike_times : array-like
        Sorted spike/event times.

    Returns
    -------
    DescriptiveResult
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_times = np.sort(spike_times)

    if len(spike_times) < 2:
        raise ValueError("Need at least 2 spike times for ISI analysis.")

    isi = np.diff(spike_times)
    mean_isi = float(np.mean(isi))
    std_isi = float(np.std(isi))
    cv = std_isi / mean_isi if mean_isi > 0 else 0.0
    median_isi = float(np.median(isi))
    min_isi = float(np.min(isi))
    max_isi = float(np.max(isi))
    mean_rate = 1.0 / mean_isi if mean_isi > 0 else 0.0

    return DescriptiveResult(
        name="isi_analyze",
        value=mean_isi,
        extra={
            "isi": isi,
            "mean": mean_isi,
            "std": std_isi,
            "cv": cv,
            "median": median_isi,
            "min": min_isi,
            "max": max_isi,
            "mean_rate": mean_rate,
            "n_spikes": len(spike_times),
            "n_intervals": len(isi),
        },
    )


isisz = isi_analyze


def cheatsheet() -> str:
    return "isi_analyze({}) -> Inter-spike interval analysis."
