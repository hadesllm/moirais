# morie.fn -- function file (hadesllm/morie)
"""Myopulse percentage rate of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def myopulse_rate_fn(x: np.ndarray, threshold: float | None = None) -> DescriptiveResult:
    """Compute the myopulse percentage rate (MYOP).

    'Hope is not a strategy.' -- Admiral Ackbar
    """
    from morie._waveform import myopulse_rate as _backend

    result = _backend(x, threshold=threshold)
    return DescriptiveResult(name="myopulse_rate", value=float(result))


alias = myopulse_rate_fn


def cheatsheet() -> str:
    return "myopulse_rate_fn({}) -> Myopulse percentage rate of a signal."
