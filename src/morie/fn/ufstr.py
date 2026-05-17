"""Unfolding stress computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def unfolding_stress(X_r, X_s, D, weights=None) -> DescriptiveResult:
    """Compute raw stress for an unfolding configuration.

    :param X_r: Row (respondent) coordinates.
    :param X_s: Column (stimulus) coordinates.
    :param D: Observed distance matrix.
    :param weights: Optional weight matrix.
    :return: DescriptiveResult with stress value.

    .. epigraph:: Logic is the foundation of all certain knowledge. -- Leonhard Euler
    """
    from morie._spatial_voting import unfolding_stress as _fn

    result = _fn(X_r, X_s, D, weights=weights)
    return DescriptiveResult(
        name="unfolding_stress",
        value=result,
        extra={"stress": result},
    )


ufstr = unfolding_stress


def cheatsheet() -> str:
    return "unfolding_stress({}) -> Unfolding stress computation."
