"""Root mean square value."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The happiness of your life depends upon the quality of your thoughts. -- Marcus Aurelius"


def rms_value(x, **kwargs) -> DescriptiveResult:
    r"""Compute the root mean square (RMS) value of signal *x*.

    .. math::

        \\text{RMS} = \\sqrt{\\frac{1}{N} \\sum_{n=0}^{N-1} x^2(n)}

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x**2)))
    return DescriptiveResult(
        name="rms_value",
        value=rms,
        extra={"rms": rms, "n": len(x)},
    )


srms = rms_value


def cheatsheet() -> str:
    return "rms_value({}) -> Root mean square value."
