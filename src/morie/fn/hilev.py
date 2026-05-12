# morie.fn -- function file (hadesllm/morie)
"""Analytic signal envelope via Hilbert transform."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ._containers import DescriptiveResult

_QUOTE = "The ability to speak does not make you intelligent."


def hilbert_envelope(x, **kwargs) -> DescriptiveResult:
    """Compute signal envelope using the Hilbert transform.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    return DescriptiveResult(
        name="hilbert_envelope",
        value=float(np.max(envelope)),
        extra={
            "envelope": envelope,
            "analytic_signal": analytic,
            "instantaneous_phase": inst_phase,
        },
    )


hilev = hilbert_envelope


def cheatsheet() -> str:
    return "hilbert_envelope({}) -> Analytic signal envelope via Hilbert transform."
