"""Peak factor."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your eyes can deceive you. Don't trust them."


def peak_factor(x, **kwargs) -> DescriptiveResult:
    r"""Compute the peak factor of signal *x*.

    .. math::

        \\text{PF} = \\frac{\\max(|x|)}{\\text{RMS}(x)}

    Equivalent to crest factor under alternative naming.

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
    peak = float(np.max(np.abs(x)))
    pf = peak / rms if rms > 0 else float("inf")
    return DescriptiveResult(
        name="peak_factor",
        value=pf,
        extra={"peak_factor": pf, "peak": peak, "rms": rms, "n": len(x)},
    )


spkfc = peak_factor


def cheatsheet() -> str:
    return "peak_factor({}) -> Peak factor."
