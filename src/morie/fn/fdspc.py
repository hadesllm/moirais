# morie.fn — function file (hadesllm/morie)
"""Fractal dimension from power spectral density."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def fractal_dim_from_psd(psd: np.ndarray, freqs: np.ndarray) -> DescriptiveResult:
    """All models are wrong, but some are useful. — George E. P. Box"""
    from morie._spectral import fractal_dim_from_psd as _backend

    fd = _backend(psd, freqs)
    return DescriptiveResult(
        name="fractal_dim_psd",
        value=fd,
        extra={"fd": fd},
    )


fdspc = fractal_dim_from_psd


def cheatsheet() -> str:
    return "fractal_dim_from_psd({}) -> Fractal dimension from power spectral density."
