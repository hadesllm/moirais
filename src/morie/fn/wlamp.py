"""Willison amplitude of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def willison_amp(x: np.ndarray, threshold: float | None = None) -> DescriptiveResult:
    """Knowledge itself is power. — Francis Bacon"""
    from morie._waveform import willison_amplitude as _backend

    result = _backend(x, threshold=threshold)
    return DescriptiveResult(name="willison_amplitude", value=int(result))


alias = willison_amp


def cheatsheet() -> str:
    return "willison_amp({}) -> Willison amplitude of a signal."
