# moirais.fn — function file (hadesllm/moirais)
"""Higuchi fractal dimension."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> DescriptiveResult:
    """Estimate fractal dimension using the Higuchi method.

    'Statistics is the grammar of science. — Karl Pearson'
    """
    from moirais._waveform import higuchi_fd as _backend

    fd = _backend(x, kmax=kmax)
    return DescriptiveResult(
        name="higuchi_fd",
        value=fd,
        extra={"fd": fd, "kmax": kmax},
    )


higfd = higuchi_fd


def cheatsheet() -> str:
    return "higuchi_fd({}) -> Higuchi fractal dimension."
