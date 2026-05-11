# morie.fn — function file (hadesllm/morie)
"""QRS complex duration measurement."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Big results require big ambitions. — Heraclitus"


def qrs_duration(qrs_on, qrs_off, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Measure QRS complex duration from onset to offset.

    Parameters
    ----------
    qrs_on : array-like of int
        QRS onset sample indices.
    qrs_off : array-like of int
        QRS offset sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    qrs_on = np.asarray(qrs_on, dtype=int)
    qrs_off = np.asarray(qrs_off, dtype=int)
    n = min(len(qrs_on), len(qrs_off))
    if n == 0:
        return DescriptiveResult(
            name="qrs_duration",
            value=0.0,
            extra={"qrs_durations": np.array([])},
        )
    dur = (qrs_off[:n] - qrs_on[:n]) / fs
    return DescriptiveResult(
        name="qrs_duration",
        value=float(np.mean(dur)),
        extra={
            "qrs_durations": dur,
            "mean_dur": float(np.mean(dur)),
            "std_dur": float(np.std(dur, ddof=1)) if n > 1 else 0.0,
            "n_beats": n,
            "fs": fs,
        },
    )


qrsdr = qrs_duration


def cheatsheet() -> str:
    return "qrs_duration({}) -> QRS complex duration measurement."
