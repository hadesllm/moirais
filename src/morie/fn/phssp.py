# morie.fn — function file (hadesllm/morie)
"""Phase spectrum."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Patience is bitter, but its fruit is sweet. — Aristotle"


def phase_spectrum(x, **kwargs) -> DescriptiveResult:
    """Compute the phase spectrum of signal *x*.

    .. math::

        \\angle X(k) = \\text{atan2}(\\text{Im}(X(k)), \\text{Re}(X(k)))

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    X = np.fft.fft(x)
    phase = np.angle(X)
    return DescriptiveResult(
        name="phase_spectrum",
        value=float(np.max(np.abs(phase))),
        extra={"phase": phase, "N": len(x)},
    )


phssp = phase_spectrum


def cheatsheet() -> str:
    return "phase_spectrum({}) -> Phase spectrum."
