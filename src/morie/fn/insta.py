# morie.fn -- function file (hadesllm/morie)
"""Instantaneous amplitude (envelope) via Hilbert transform."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ._containers import DescriptiveResult

_QUOTE = "The garbage will do!"


def instantaneous_amp(x: np.ndarray) -> DescriptiveResult:
    r"""Instantaneous amplitude (envelope) via the analytic signal.

    .. math::

        A(t) = |z(t)| = \\sqrt{x^2(t) + \\hat{x}^2(t)}

    where :math:`\\hat{x}` is the Hilbert transform of *x*.

    Parameters
    ----------
    x : array-like
        1-D input signal.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``envelope``, ``mean_amplitude``, ``peak_amplitude``.
    """
    x = np.asarray(x, dtype=float).ravel()
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    return DescriptiveResult(
        name="instantaneous_amp",
        value=float(np.mean(envelope)),
        extra={
            "envelope": envelope,
            "mean_amplitude": float(np.mean(envelope)),
            "peak_amplitude": float(np.max(envelope)),
        },
    )


insta = instantaneous_amp


def cheatsheet() -> str:
    return "instantaneous_amp({}) -> Instantaneous amplitude (envelope) via Hilbert transform."
