# moirais.fn — function file (hadesllm/moirais)
"""Onset detection via energy envelope."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def onset_detect_fn(
    x: np.ndarray, fs: float = 1000.0, energy_window_ms: float = 20.0, threshold_factor: float = 3.0
) -> DescriptiveResult:
    """Detect signal onsets using energy envelope thresholding.

    'An unexamined life is not worth living. — Socrates'
    """
    from moirais._detection import onset_detect as _backend

    onsets = _backend(x, fs, energy_window_ms=energy_window_ms, threshold_factor=threshold_factor)
    return DescriptiveResult(
        name="onset_detect",
        value=int(len(onsets)),
        extra={"onsets": onsets},
    )


alias = onset_detect_fn


def cheatsheet() -> str:
    return "onset_detect_fn({}) -> Onset detection via energy envelope."
