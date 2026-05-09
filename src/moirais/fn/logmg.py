# moirais.fn — function file (hadesllm/moirais)
"""Log magnitude spectrum."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Errors using inadequate data are much less than those using no data at all. — Charles Babbage"


def log_magnitude_spectrum(x, **kwargs) -> DescriptiveResult:
    """Compute the log magnitude spectrum in decibels.

    .. math::

        20 \\cdot \\log_{10}(|X(k)|) \\; \\text{dB}

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
    mag = np.abs(X)
    mag = np.where(mag == 0, np.finfo(float).tiny, mag)
    log_mag = 20.0 * np.log10(mag)
    return DescriptiveResult(
        name="log_magnitude_spectrum",
        value=float(np.max(log_mag)),
        extra={"log_magnitude_db": log_mag, "N": len(x)},
    )


logmg = log_magnitude_spectrum


def cheatsheet() -> str:
    return "log_magnitude_spectrum({}) -> Log magnitude spectrum."
