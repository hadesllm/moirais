# morie.fn — function file (hadesllm/morie)
"""PolarQuant decoder: reconstruct from magnitude + direction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polarquant_decode(
    magnitude: float,
    direction: np.ndarray,
) -> DescriptiveResult:
    """PolarQuant decoding: reconstruct vector from magnitude and direction.

    :param magnitude: Scalar magnitude.
    :param direction: Unit direction vector.
    :return: DescriptiveResult with reconstructed vector.
    """
    direction = np.asarray(direction, dtype=np.float64).ravel()
    reconstructed = magnitude * direction
    return DescriptiveResult(
        name="polarquant_decode",
        value=float(np.linalg.norm(reconstructed)),
        extra={
            "reconstructed": reconstructed,
            "magnitude": magnitude,
        },
    )


def cheatsheet() -> str:
    return "polarquant_decode(magnitude, direction) -> PolarQuant reconstruct"


pqdec = polarquant_decode
