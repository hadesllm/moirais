# morie.fn — function file (hadesllm/morie)
"""Exponential moving average."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def exponential_ma(x, alpha=0.3, **kwargs) -> DescriptiveResult:
    """Apply an exponential moving average filter.

    .. math::

        y(n) = \\alpha \\cdot x(n) + (1 - \\alpha) \\cdot y(n-1)

    Parameters
    ----------
    x : array-like
        Input signal.
    alpha : float
        Smoothing factor in (0, 1].

    Returns
    -------
    DescriptiveResult
        ``value`` is the smoothed signal (ndarray).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return DescriptiveResult(
        name="exponential_ma",
        value=out,
        extra={"alpha": alpha, "n": len(x)},
    )


emavg = exponential_ma


def cheatsheet() -> str:
    return "exponential_ma({}) -> Exponential moving average."
