# morie.fn -- function file (hadesllm/morie)
"""Root-mean-square of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rms_signal(x: np.ndarray) -> DescriptiveResult:
    """Compute the root-mean-square (RMS) amplitude.

    'Rebellions are built on hope.' -- Jyn Erso
    """
    from morie._waveform import rms as _backend

    result = _backend(x)
    return DescriptiveResult(name="rms", value=float(result))


alias = rms_signal


def cheatsheet() -> str:
    return "rms_signal({}) -> Root-mean-square of a signal."
