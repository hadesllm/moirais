# morie.fn -- function file (hadesllm/morie)
"""DC component removal."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "These aren't the droids you're looking for."


def dc_removal(x) -> SignalResult:
    """Remove the DC component from signal *x*.

    .. math::

        y(n) = x(n) - \\bar{x}

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    dc = float(np.mean(x))
    y = x - dc
    return SignalResult(
        name="dc_removal",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"dc_value": dc},
    )


dcsub = dc_removal


def cheatsheet() -> str:
    return "dc_removal({}) -> DC component removal."
