# morie.fn — function file (hadesllm/morie)
"""PolarQuant encoder: separate magnitude + direction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polarquant_encode(x: np.ndarray) -> DescriptiveResult:
    """PolarQuant encoding: decompose vector into magnitude and direction.

    :param x: Input vector (1-D).
    :return: DescriptiveResult with magnitude scalar and unit direction.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    magnitude = float(np.linalg.norm(x))
    if magnitude > 0:
        direction = x / magnitude
    else:
        direction = np.zeros_like(x)
    return DescriptiveResult(
        name="polarquant_encode",
        value=magnitude,
        extra={
            "magnitude": magnitude,
            "direction": direction,
            "d": len(x),
        },
    )


def cheatsheet() -> str:
    return "polarquant_encode(x) -> PolarQuant: separate magnitude + direction"


pqenc = polarquant_encode
