"""Flat torus from rectangle identification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def flat_torus(
    a: float = 1.0,
    b: float = 1.0,
    n: int = 50,
) -> DescriptiveResult:
    """Generate a flat torus by identifying opposite sides of [0,a] x [0,b].

    Returns a grid of points on the fundamental domain.
    Distances wrap: d(x1,x2) = min(|x1-x2|, a-|x1-x2|).

    :param a: Width of rectangle.
    :param b: Height of rectangle.
    :param n: Grid resolution per side.
    :return: DescriptiveResult with grid coordinates and metric info.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Dimensions must be positive, got a={a}, b={b}.")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")
    xs = np.linspace(0, a, n, endpoint=False)
    ys = np.linspace(0, b, n, endpoint=False)
    grid_x, grid_y = np.meshgrid(xs, ys)
    area = a * b
    return DescriptiveResult(
        name="flat_torus",
        value=area,
        extra={
            "grid_x": grid_x,
            "grid_y": grid_y,
            "a": a,
            "b": b,
            "n": n,
            "area": area,
            "curvature": 0.0,
        },
    )


def cheatsheet() -> str:
    return "flat_torus(a, b, n) -> flat torus from rectangle identification"
