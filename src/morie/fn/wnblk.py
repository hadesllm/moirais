"""Blackman window."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Power! Unlimited power!"


def blackman_window(N: int, **kwargs) -> DescriptiveResult:
    r"""Generate a Blackman window of length *N*.

    .. math::

        w(n) = 0.42 - 0.5\\cos\\frac{2\\pi n}{N-1}
               + 0.08\\cos\\frac{4\\pi n}{N-1}

    Parameters
    ----------
    N : int
        Window length.

    Returns
    -------
    DescriptiveResult
    """
    n = np.arange(N)
    if N > 1:
        w = 0.42 - 0.5 * np.cos(2.0 * np.pi * n / (N - 1)) + 0.08 * np.cos(4.0 * np.pi * n / (N - 1))
    else:
        w = np.ones(1)
    return DescriptiveResult(
        name="blackman_window",
        value=float(N),
        extra={"window": w, "N": N},
    )


wnblk = blackman_window


def cheatsheet() -> str:
    return "blackman_window({}) -> Blackman window."
