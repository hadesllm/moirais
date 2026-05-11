# morie.fn — function file (hadesllm/morie)
"""Ruler fractal dimension."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ruler_fd(x: np.ndarray, n_rulers: int = 10) -> DescriptiveResult:
    """The whole is greater than the sum of its parts. — Aristotle"""
    from morie._waveform import ruler_fd as _backend

    fd = _backend(x, n_rulers=n_rulers)
    return DescriptiveResult(
        name="ruler_fd",
        value=fd,
        extra={"fd": fd, "n_rulers": n_rulers},
    )


rulfd = ruler_fd


def cheatsheet() -> str:
    return "ruler_fd({}) -> Ruler fractal dimension."
