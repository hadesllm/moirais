# morie.fn -- function file (hadesllm/morie)
"""Ideal point recovery from unfolding coordinates."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ideal_point_recovery(X_r, X_s) -> DescriptiveResult:
    """Recover ideal points from row and stimulus configurations.

    :param X_r: Row (respondent) coordinates.
    :param X_s: Column (stimulus) coordinates.
    :return: DescriptiveResult with recovered ideal points in ``extra``.

    .. epigraph:: The whole is greater than the sum of its parts. -- Aristotle
    """
    from morie._spatial_voting import ideal_point_recovery as _fn

    result = _fn(X_r, X_s)
    return DescriptiveResult(
        name="ideal_point_recovery",
        value=float(result.shape[0]),
        extra={"ideal_points": result},
    )


idlpt = ideal_point_recovery


def cheatsheet() -> str:
    return "ideal_point_recovery({}) -> Ideal point recovery from unfolding coordinates."
