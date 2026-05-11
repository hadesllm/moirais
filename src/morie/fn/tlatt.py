"""Torus lattice from basis vectors."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_lattice(
    a: float = 1.0,
    b: float = 1.0,
    angle: float = 90.0,
    n: int = 5,
) -> DescriptiveResult:
    """Generate a 2D lattice for a flat torus from basis vectors.

    Basis vectors: e1 = (a, 0), e2 = (b*cos(angle), b*sin(angle)).
    Returns lattice points m*e1 + n*e2 for m, n in [-n, n].

    :param a: Length of first basis vector.
    :param b: Length of second basis vector.
    :param angle: Angle between basis vectors in degrees.
    :param n: Range of lattice indices.
    :return: DescriptiveResult with lattice points and area.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Basis lengths must be positive, got a={a}, b={b}.")
    theta = np.radians(angle)
    e1 = np.array([a, 0.0])
    e2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    area = abs(a * b * np.sin(theta))
    ms = np.arange(-n, n + 1)
    ns = np.arange(-n, n + 1)
    mm, nn = np.meshgrid(ms, ns)
    points_x = mm * e1[0] + nn * e2[0]
    points_y = mm * e1[1] + nn * e2[1]
    return DescriptiveResult(
        name="torus_lattice",
        value=area,
        extra={
            "points_x": points_x.ravel(),
            "points_y": points_y.ravel(),
            "e1": e1.tolist(),
            "e2": e2.tolist(),
            "area": area,
            "n_points": len(points_x.ravel()),
        },
    )


def cheatsheet() -> str:
    return "torus_lattice(a, b, angle, n) -> lattice on flat torus"
