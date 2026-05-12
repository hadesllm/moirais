# morie.fn -- function file (hadesllm/morie)
"""Integrated EMG (IEMG) of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def integrated_emg_fn(x: np.ndarray) -> DescriptiveResult:
    """Time discovers truth. -- Seneca"""
    from morie._waveform import integrated_emg as _backend

    result = _backend(x)
    return DescriptiveResult(name="integrated_emg", value=float(result))


alias = integrated_emg_fn


def cheatsheet() -> str:
    return "integrated_emg_fn({}) -> Integrated EMG (IEMG) of a signal."
