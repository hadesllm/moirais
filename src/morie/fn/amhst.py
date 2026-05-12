# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Amplitude histogram of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def amplitude_hist(x: np.ndarray, n_bins: int = 50) -> DescriptiveResult:
    """Our greatest glory is not in never falling, but in rising every time we fall. -- Confucius"""
    from morie._waveform import amplitude_histogram as _backend

    hist_dict = _backend(x, n_bins=n_bins)
    return DescriptiveResult(
        name="amplitude_histogram",
        value=None,
        extra=hist_dict,
    )


alias = amplitude_hist


def cheatsheet() -> str:
    return "amplitude_hist({}) -> Amplitude histogram of a signal."
