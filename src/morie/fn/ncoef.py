# morie.fn -- function file (hadesllm/morie)
"""Normalize coefficient vector to unit normal."""

from __future__ import annotations

from ._containers import DescriptiveResult


def normalize_coefficients(beta) -> DescriptiveResult:
    """Normalize a coefficient vector to unit length.

    :param beta: Coefficient vector.
    :return: DescriptiveResult with unit normal vector.

    .. epigraph:: Measure what is measurable, and make measurable what is not. -- Galileo Galilei
    """
    import numpy as np

    b = np.asarray(beta, dtype=float).ravel()
    norm = float(np.linalg.norm(b))
    if norm > 0:
        unit = b / norm
    else:
        unit = b.copy()
    return DescriptiveResult(
        name="normalize_coefficients",
        value=norm,
        extra={"unit_vector": unit.tolist(), "original_norm": norm},
    )


ncoef = normalize_coefficients


def cheatsheet() -> str:
    return "normalize_coefficients({}) -> Normalize coefficient vector to unit normal."
