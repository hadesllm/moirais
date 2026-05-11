"""Zero crossing rate."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this."


def zero_crossing_rate(x, fs=1.0, **kwargs) -> DescriptiveResult:
    """Compute the zero crossing rate of signal *x*.

    .. math::

        \\text{ZCR} = \\frac{1}{N-1} \\sum_{n=1}^{N-1}
        \\frac{|\\text{sign}(x[n]) - \\text{sign}(x[n-1])|}{2}

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency (Hz). Used to report crossings per second.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    signs = np.sign(x)
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    zcr = crossings / max(len(x) - 1, 1)
    return DescriptiveResult(
        name="zero_crossing_rate",
        value=float(zcr),
        extra={"zcr": float(zcr), "crossings": crossings, "rate_per_sec": float(zcr * fs), "n": len(x)},
    )


szcr = zero_crossing_rate


def cheatsheet() -> str:
    return "zero_crossing_rate({}) -> Zero crossing rate."
