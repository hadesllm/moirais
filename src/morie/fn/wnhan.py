"""Hanning window."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def hanning_window(N: int, **kwargs) -> DescriptiveResult:
    """Generate a Hanning (Hann) window of length *N*.

    .. math::

        w(n) = 0.5 \\left(1 - \\cos\\frac{2\\pi n}{N-1}\\right)

    Parameters
    ----------
    N : int
        Window length.

    Returns
    -------
    DescriptiveResult
    """
    n = np.arange(N)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (N - 1))) if N > 1 else np.ones(1)
    return DescriptiveResult(
        name="hanning_window",
        value=float(N),
        extra={"window": w, "N": N},
    )


wnhan = hanning_window


def cheatsheet() -> str:
    return "hanning_window({}) -> Hanning window."
