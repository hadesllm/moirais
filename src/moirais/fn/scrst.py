# moirais.fn — function file (hadesllm/moirais)
"""Crest factor."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It's a trap!"


def crest_factor(x, **kwargs) -> DescriptiveResult:
    """Compute the crest factor of signal *x*.

    .. math::

        \\text{CF} = \\frac{\\max(|x|)}{\\text{RMS}(x)}

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
    cf = peak / rms if rms > 0 else float("inf")
    return DescriptiveResult(
        name="crest_factor",
        value=cf,
        extra={"crest_factor": cf, "peak": peak, "rms": rms, "n": len(x)},
    )


scrst = crest_factor


def cheatsheet() -> str:
    return "crest_factor({}) -> Crest factor."
