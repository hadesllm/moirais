"""Threshold-based peak/event detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def threshold_detect(
    x: np.ndarray, threshold: float, min_distance: int = 1, direction: str = "above"
) -> DescriptiveResult:
    """Detect indices where signal crosses a threshold.

    'That which does not kill us makes us stronger. -- Friedrich Nietzsche'
    """
    from morie._detection import threshold_detect as _backend

    indices = _backend(x, threshold, min_distance=min_distance, direction=direction)
    return DescriptiveResult(
        name="threshold_detect",
        value=int(len(indices)),
        extra={"indices": indices},
    )


alias = threshold_detect


def cheatsheet() -> str:
    return "threshold_detect({}) -> Threshold-based peak/event detection."
