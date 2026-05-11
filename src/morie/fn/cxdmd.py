# morie.fn — function file (hadesllm/morie)
"""Complex demodulation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def complex_demodulation(x: np.ndarray, fc: float, fs: float = 1.0) -> DescriptiveResult:
    """Extract amplitude envelope and instantaneous phase via complex demodulation.

    'We are what they grow beyond.'
    """
    from morie._waveform import complex_demodulation as _backend

    envelope, phase = _backend(x, fc=fc, fs=fs)
    return DescriptiveResult(
        name="complex_demodulation",
        value=len(envelope),
        extra={"envelope": envelope, "phase": phase, "fc": fc, "fs": fs},
    )


cxdmd = complex_demodulation


def cheatsheet() -> str:
    return "complex_demodulation({}) -> Complex demodulation."
