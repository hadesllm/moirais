"""Zero-crossing rate of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def zero_crossing_rate(x: np.ndarray, frame_length: int | None = None) -> DescriptiveResult:
    """Compute zero-crossing rate of a signal.

    'Character is destiny. — Heraclitus'
    """
    from morie._detection import zero_crossing_rate as _backend

    rate = _backend(x, frame_length=frame_length)
    if isinstance(rate, np.ndarray):
        value = float(np.mean(rate))
        extra = {"per_frame": rate}
    else:
        value = float(rate)
        extra = {}
    return DescriptiveResult(
        name="zero_crossing_rate",
        value=value,
        extra=extra,
    )


alias = zero_crossing_rate


def cheatsheet() -> str:
    return "zero_crossing_rate({}) -> Zero-crossing rate of a signal."
