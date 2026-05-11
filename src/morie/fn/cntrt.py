# morie.fn — function file (hadesllm/morie)
"""Centroidal time of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def centroidal_time(x: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Compute the energy-weighted centroidal time.

    'Your focus determines your reality.' — Qui-Gon Jinn
    """
    from morie._waveform import centroidal_time as _backend

    ct = _backend(x, fs=fs)
    return DescriptiveResult(
        name="centroidal_time",
        value=ct,
        extra={"centroidal_time": ct, "fs": fs},
    )


cntrt = centroidal_time


def cheatsheet() -> str:
    return "centroidal_time({}) -> Centroidal time of a signal."
