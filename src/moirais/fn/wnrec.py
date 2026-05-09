"""Rectangular window."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You have power over your mind — not outside events. — Marcus Aurelius"


def rectangular_window(N: int, **kwargs) -> DescriptiveResult:
    """Generate a rectangular window of length *N*.

    .. math::

        w(n) = 1 \\quad \\text{for } 0 \\le n \\le N-1

    Parameters
    ----------
    N : int
        Window length.

    Returns
    -------
    DescriptiveResult
    """
    w = np.ones(N)
    return DescriptiveResult(
        name="rectangular_window",
        value=float(N),
        extra={"window": w, "N": N},
    )


wnrec = rectangular_window


def cheatsheet() -> str:
    return "rectangular_window({}) -> Rectangular window."
