# morie.fn -- function file (hadesllm/morie)
"""Duty cycle."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I am the Senate."


def duty_cycle(x, threshold=0.0, **kwargs) -> DescriptiveResult:
    """Compute the duty cycle of signal *x*.

    Fraction of samples above *threshold*.

    Parameters
    ----------
    x : array-like
        Input signal.
    threshold : float
        Level to compare against.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    above = int(np.sum(x > threshold))
    dc = above / len(x) if len(x) > 0 else 0.0
    return DescriptiveResult(
        name="duty_cycle",
        value=float(dc),
        extra={"duty_cycle": float(dc), "above_count": above, "threshold": threshold, "n": len(x)},
    )


sduty = duty_cycle


def cheatsheet() -> str:
    return "duty_cycle({}) -> Duty cycle."
