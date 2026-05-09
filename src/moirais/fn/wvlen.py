"""Waveform length (arc length) of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def waveform_length_fn(x: np.ndarray) -> DescriptiveResult:
    """Compute the waveform length (sum of absolute first differences).

    'I sense great fear in you.'
    """
    from moirais._waveform import waveform_length as _backend

    result = _backend(x)
    return DescriptiveResult(name="waveform_length", value=float(result))


alias = waveform_length_fn


def cheatsheet() -> str:
    return "waveform_length_fn({}) -> Waveform length (arc length) of a signal."
