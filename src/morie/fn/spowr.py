"""Signal power."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies, with thunderous applause."


def signal_power(x, **kwargs) -> DescriptiveResult:
    r"""Compute the average power of signal *x*.

    .. math::

        P = \\frac{1}{N} \\sum_{n=0}^{N-1} |x(n)|^2

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    power = float(np.mean(np.abs(x) ** 2))
    return DescriptiveResult(
        name="signal_power",
        value=power,
        extra={"power": power, "n": len(x)},
    )


spowr = signal_power


def cheatsheet() -> str:
    return "signal_power({}) -> Signal power."
