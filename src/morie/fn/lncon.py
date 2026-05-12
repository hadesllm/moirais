# morie.fn -- function file (hadesllm/morie)
"""Linear convolution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The garbage will do!"


def linear_convolution(x, h, **kwargs) -> DescriptiveResult:
    r"""Compute linear convolution of *x* and *h*.

    .. math::

        y(n) = \\sum_{k} x(k) \\cdot h(n - k)

    Parameters
    ----------
    x : array-like
        Input signal.
    h : array-like
        Impulse response / filter.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.convolve(x, h)
    return DescriptiveResult(
        name="linear_convolution",
        value=float(np.max(np.abs(y))),
        extra={"output": y, "len_x": len(x), "len_h": len(h)},
    )


lncon = linear_convolution


def cheatsheet() -> str:
    return "linear_convolution({}) -> Linear convolution."
