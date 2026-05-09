"""Turns count of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def turns_count_fn(x: np.ndarray, threshold: float = 0.0) -> DescriptiveResult:
    """Count the number of turns (direction changes) in a signal.

    'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius'
    """
    from moirais._waveform import turns_count as _backend

    result = _backend(x, threshold=threshold)
    return DescriptiveResult(name="turns_count", value=int(result))


alias = turns_count_fn


def cheatsheet() -> str:
    return "turns_count_fn({}) -> Turns count of a signal."
