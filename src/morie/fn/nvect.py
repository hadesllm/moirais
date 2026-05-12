# morie.fn -- function file (hadesllm/morie)
"""Normal vector projection onto recovered space."""

from __future__ import annotations

from ._containers import DescriptiveResult


def normal_vector_projection(
    ideal_points,
    external_measure,
) -> DescriptiveResult:
    """Project external measures onto latent space (Eqs 2.12-2.13).

    :param ideal_points: (n x dims) ideal point coordinates.
    :param external_measure: (n,) external variable.
    :return: DescriptiveResult with normal vector and angle.

    .. epigraph:: "The truth is rarely pure and never simple." -- Picard, Star Trek
    """
    from morie._spatial_voting import normal_vectors as _fn

    result = _fn(ideal_points, external_measure)
    return DescriptiveResult(
        name="normal_vector_projection",
        value=result["r_squared"],
        extra=result,
    )


nvect = normal_vector_projection


def cheatsheet() -> str:
    return "normal_vector_projection({}) -> Normal vector projection onto recovered space."
