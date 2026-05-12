# morie.fn -- function file (hadesllm/morie)
"""Instantaneous frequency via analytic signal."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful the mind of a child is."


def instantaneous_freq(x, fs: float = 1.0) -> DescriptiveResult:
    """Compute instantaneous frequency via the analytic signal.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    analytic = hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    amp = np.abs(analytic)
    return DescriptiveResult(
        name="instantaneous_freq",
        value=float(np.mean(inst_freq)),
        extra={"inst_freq": inst_freq, "amplitude": amp, "phase": phase, "fs": fs},
    )


insft = instantaneous_freq


def cheatsheet() -> str:
    return "instantaneous_freq({}) -> Instantaneous frequency via analytic signal."
