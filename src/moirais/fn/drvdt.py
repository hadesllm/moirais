# moirais.fn — function file (hadesllm/moirais)
"""Derivative-based peak detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def derivative_detect(x: np.ndarray, fs: float = 1.0, threshold_factor: float = 0.5) -> DescriptiveResult:
    """Detect peaks via first-derivative zero crossings.

    'Errors using inadequate data are much less than those using no data at all. — Charles Babbage'
    """
    from moirais._detection import derivative_detect as _backend

    peaks = _backend(x, fs=fs, threshold_factor=threshold_factor)
    return DescriptiveResult(
        name="derivative_detect",
        value=int(len(peaks)),
        extra={"peaks": peaks},
    )


alias = derivative_detect


def cheatsheet() -> str:
    return "derivative_detect({}) -> Derivative-based peak detection."
