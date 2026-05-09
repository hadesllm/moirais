"""Mean crossing rate."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Let the past die. Kill it, if you have to."


def mean_crossing_rate(x, fs=1.0, **kwargs) -> DescriptiveResult:
    """Compute the mean crossing rate of signal *x*.

    Like the zero crossing rate but computed around the signal mean
    rather than zero.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    signs = np.sign(centered)
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    mcr = crossings / max(len(x) - 1, 1)
    return DescriptiveResult(
        name="mean_crossing_rate",
        value=float(mcr),
        extra={"mcr": float(mcr), "crossings": crossings, "rate_per_sec": float(mcr * fs), "n": len(x)},
    )


smcr = mean_crossing_rate


def cheatsheet() -> str:
    return "mean_crossing_rate({}) -> Mean crossing rate."
