"""Torus surface area and volume."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_surface(R: float = 3.0, r: float = 1.0) -> DescriptiveResult:
    """Surface area and volume of a torus with major radius *R* and minor radius *r*.

    Surface area: :math:`S = 4\\pi^2 R r`
    Volume: :math:`V = 2\\pi^2 R r^2`

    :param R: Major radius (centre of tube to centre of torus). Must be > 0.
    :param r: Minor radius (radius of tube). Must be > 0 and <= R.
    :return: DescriptiveResult with surface_area and volume in *extra*.
    :raises ValueError: If R or r non-positive or r > R.
    """
    if R <= 0 or r <= 0:
        raise ValueError(f"Radii must be positive, got R={R}, r={r}.")
    if r > R:
        raise ValueError(f"Minor radius r={r} must be <= major radius R={R}.")
    surface_area = 4.0 * np.pi**2 * R * r
    volume = 2.0 * np.pi**2 * R * r**2
    return DescriptiveResult(
        name="torus_surface",
        value=surface_area,
        extra={"surface_area": surface_area, "volume": volume, "R": R, "r": r},
    )


def cheatsheet() -> str:
    return "torus_surface(R, r) -> surface area & volume of torus"
