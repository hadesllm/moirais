# moirais.fn — function file (hadesllm/moirais)
"""Chirp (frequency sweep) detection via instantaneous frequency."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

from ._containers import DescriptiveResult

_QUOTE = "I have the high ground."


def chirp_detect(
    x: np.ndarray,
    fs: float = 1.0,
) -> DescriptiveResult:
    """Detect chirp (frequency sweep) in a signal.

    Estimates linear frequency trend from instantaneous frequency.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``chirp_rate`` (Hz/s), ``r_squared``,
        ``f_start``, ``f_end``, ``inst_freq``.
    """
    x = np.asarray(x, dtype=float).ravel()
    analytic = hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    t = np.arange(len(inst_freq)) / fs
    slope, intercept, r, _, _ = linregress(t, inst_freq)
    return DescriptiveResult(
        name="chirp_detect",
        value=float(slope),
        extra={
            "chirp_rate": float(slope),
            "r_squared": float(r**2),
            "f_start": float(intercept),
            "f_end": float(intercept + slope * t[-1]),
            "inst_freq": inst_freq,
        },
    )


chird = chirp_detect


def cheatsheet() -> str:
    return "chirp_detect({}) -> Chirp (frequency sweep) detection via instantaneous frequenc"
