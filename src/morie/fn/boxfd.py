# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Box-counting fractal dimension."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def box_counting_fd(x: np.ndarray, n_scales: int = 10) -> DescriptiveResult:
    """Study the past if you would define the future. -- Confucius"""
    from morie._waveform import box_counting_fd as _backend

    fd = _backend(x, n_scales=n_scales)
    return DescriptiveResult(
        name="box_counting_fd",
        value=fd,
        extra={"fd": fd, "n_scales": n_scales},
    )


boxfd = box_counting_fd


def cheatsheet() -> str:
    return "box_counting_fd({}) -> Box-counting fractal dimension."
