# moirais.fn — function file (hadesllm/moirais)
"""Instantaneous phase via Hilbert transform."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we."


def instantaneous_phase(x: np.ndarray) -> DescriptiveResult:
    """Instantaneous phase via analytic signal.

    .. math::

        \\phi(t) = \\arctan\\!\\left(\\frac{\\hat{x}(t)}{x(t)}\\right)

    Parameters
    ----------
    x : array-like
        1-D input signal.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``phase`` (unwrapped), ``phase_wrapped``.
    """
    x = np.asarray(x, dtype=float).ravel()
    analytic = hilbert(x)
    phase_wrapped = np.angle(analytic)
    phase = np.unwrap(phase_wrapped)
    return DescriptiveResult(
        name="instantaneous_phase",
        value=float(phase[-1] - phase[0]),
        extra={"phase": phase, "phase_wrapped": phase_wrapped},
    )


instp = instantaneous_phase


def cheatsheet() -> str:
    return "instantaneous_phase({}) -> Instantaneous phase via Hilbert transform."
