"""Torus knot coordinates."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_knot(
    p: int = 2,
    q: int = 3,
    n_points: int = 500,
    R: float = 3.0,
    r: float = 1.0,
) -> DescriptiveResult:
    r"""Generate (p,q)-torus knot coordinates in R^3.

    A torus knot winds *p* times around the hole and *q* times around the tube.

    .. math::

        x = (R + r\\cos(q t))\\cos(p t), \\quad
        y = (R + r\\cos(q t))\\sin(p t), \\quad
        z = r\\sin(q t)

    :param p: Windings around the hole.
    :param q: Windings around the tube.
    :param n_points: Number of sample points.
    :param R: Major radius.
    :param r: Minor radius.
    :return: DescriptiveResult with x, y, z arrays in *extra*.
    """
    if p <= 0 or q <= 0:
        raise ValueError(f"p and q must be positive integers, got p={p}, q={q}.")
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return DescriptiveResult(
        name="torus_knot",
        value=n_points,
        extra={"x": x, "y": y, "z": z, "p": p, "q": q},
    )


def cheatsheet() -> str:
    return "torus_knot(p, q, n_points) -> (p,q)-torus knot xyz"
