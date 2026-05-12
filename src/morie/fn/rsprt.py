# morie.fn -- function file (hadesllm/morie)
"""Respiratory rate estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def respiratory_rate_fn(
    x: np.ndarray,
    fs: float = 100.0,
) -> DescriptiveResult:
    """Estimate respiratory rate from a breathing signal.

    Band-passes the signal to 0.1--0.5 Hz (6--30 breaths/min) and
    counts peaks.

    Parameters
    ----------
    x : array-like
        1-D respiratory signal (chest impedance, flow, etc.).
    fs : float
        Sampling frequency in Hz (default 100).

    Returns
    -------
    DescriptiveResult
        *value* is estimated respiratory rate in breaths per minute;
        *extra* contains ``rate_bpm``.

    References
    ----------
    Schaefer, A. & Bhatt, V. (2014). Respiratory rate monitoring.
        *Biomedical Signal Processing and Control*, 13, 44--52.
    """
    from morie._bioplot import respiratory_rate

    x = np.asarray(x, dtype=float)
    rate = respiratory_rate(x, fs=fs)
    return DescriptiveResult(
        name="respiratory_rate",
        value=rate,
        extra={"rate_bpm": rate, "fs": fs},
    )


rsprt = respiratory_rate_fn


def cheatsheet() -> str:
    return "respiratory_rate_fn({}) -> Respiratory rate estimation."
