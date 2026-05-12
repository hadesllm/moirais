# morie.fn -- function file (hadesllm/morie)
"""Parzen window probability density estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def parzen_pdf(x: np.ndarray, bandwidth: float | None = None, n_points: int = 100) -> DescriptiveResult:
    """Estimate PDF using a Parzen (Gaussian kernel) window.

    'The belonging you seek is not behind you, it is ahead.' -- Maz Kanata
    """
    from morie._waveform import parzen_pdf as _backend

    grid, density = _backend(x, bandwidth=bandwidth, n_points=n_points)
    bw = bandwidth if bandwidth is not None else 1.06 * np.std(x) * len(x) ** (-0.2)
    return DescriptiveResult(
        name="parzen_pdf",
        value=float(np.max(density)),
        extra={"grid": grid, "density": density, "bandwidth": bw},
    )


przpd = parzen_pdf


def cheatsheet() -> str:
    return "parzen_pdf({}) -> Parzen window probability density estimation."
