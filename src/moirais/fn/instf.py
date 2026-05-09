# moirais.fn — function file (hadesllm/moirais)
"""Instantaneous frequency via analytic signal."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ._containers import DescriptiveResult

_QUOTE = "Begun, the Clone War has."


def instantaneous_freq(
    x: np.ndarray,
    fs: float = 1.0,
) -> DescriptiveResult:
    """Instantaneous frequency from the analytic signal.

    .. math::

        f_i(t) = \\frac{1}{2\\pi} \\frac{d\\phi(t)}{dt}

    where :math:`\\phi(t)` is the unwrapped phase of the analytic signal.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency (default 1.0).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``inst_freq``, ``phase``, ``mean_freq``.
    """
    x = np.asarray(x, dtype=float).ravel()
    analytic = hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    mean_freq = float(np.mean(inst_freq))
    return DescriptiveResult(
        name="instantaneous_freq",
        value=mean_freq,
        extra={"inst_freq": inst_freq, "phase": phase, "mean_freq": mean_freq},
    )


instf = instantaneous_freq


def cheatsheet() -> str:
    return "instantaneous_freq({}) -> Instantaneous frequency via analytic signal."
