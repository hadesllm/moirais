# morie.fn -- function file (hadesllm/morie)
"""PolarQuant unit-sphere projection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polar_normalize(x: np.ndarray) -> DescriptiveResult:
    """Project vector onto the unit sphere (L2 normalization).

    :param x: Input vector.
    :return: DescriptiveResult with normalized vector and original norm.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(x))
    if norm > 0:
        normalized = x / norm
    else:
        normalized = np.zeros_like(x)
    return DescriptiveResult(
        name="polar_normalize",
        value=norm,
        extra={
            "normalized": normalized,
            "original_norm": norm,
            "unit_check": float(np.linalg.norm(normalized)),
        },
    )


def cheatsheet() -> str:
    return "polar_normalize(x) -> unit-sphere projection for PolarQuant"


pqnrm = polar_normalize
