# morie.fn — function file (hadesllm/morie)
"""Daubechies wavelet filter coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target."


def daubechies_coeffs(order: int = 4) -> DescriptiveResult:
    """Daubechies wavelet filter coefficients (orthogonal, compact support).

    Parameters
    ----------
    order : int
        Daubechies order (1-4 supported). Default 4.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``lo_d`` (lowpass), ``hi_d`` (highpass),
        ``lo_r`` (reconstruction lowpass), ``hi_r`` (reconstruction highpass).
    """
    from .dwtfn import _daubechies_filter

    lo_d = _daubechies_filter(order)
    hi_d = np.array([(-1) ** k * lo_d[len(lo_d) - 1 - k] for k in range(len(lo_d))])
    lo_r = lo_d[::-1]
    hi_r = hi_d[::-1]
    return DescriptiveResult(
        name="daubechies_coeffs",
        value=float(order),
        extra={
            "lo_d": lo_d,
            "hi_d": hi_d,
            "lo_r": lo_r,
            "hi_r": hi_r,
            "order": order,
            "length": len(lo_d),
        },
    )


db4fn = daubechies_coeffs


def cheatsheet() -> str:
    return "daubechies_coeffs({}) -> Daubechies wavelet filter coefficients."
