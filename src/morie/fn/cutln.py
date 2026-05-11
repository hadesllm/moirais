# morie.fn — function file (hadesllm/morie)
"""Cutting lines for Coombs mesh visualization."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cutting_line_mesh(
    normals,
    cutpoints,
    xlim: tuple = (-1.0, 1.0),
) -> DescriptiveResult:
    """Compute cutting lines for roll call visualization.

    :param normals: (n_votes x dims) normal vectors.
    :param cutpoints: (n_votes,) cutpoint offsets.
    :param xlim: x-axis limits for line endpoints.
    :return: DescriptiveResult with line endpoints and angles.

    .. epigraph:: "A sword wields no strength unless the hand that holds it has courage." -- Hero of Time, Zelda
    """
    from morie._spatial_voting import cutting_lines as _fn

    result = _fn(normals, cutpoints, xlim=xlim)
    return DescriptiveResult(
        name="cutting_line_mesh",
        value=result["n_lines"],
        extra=result,
    )


cutln = cutting_line_mesh


def cheatsheet() -> str:
    return "cutting_line_mesh({}) -> Cutting lines for Coombs mesh visualization."
