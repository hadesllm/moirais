"""Torus parametric coordinates."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_parametric(
    R: float = 3.0,
    r: float = 1.0,
    u: float | np.ndarray = 0.0,
    v: float | np.ndarray = 0.0,
) -> DescriptiveResult:
    """Parametric coordinates (x, y, z) on a torus.

    .. math::

        x = (R + r\\cos v)\\cos u, \\quad
        y = (R + r\\cos v)\\sin u, \\quad
        z = r\\sin v

    :param R: Major radius.
    :param r: Minor radius.
    :param u: Angle around the torus (radians).
    :param v: Angle around the tube (radians).
    :return: DescriptiveResult with x, y, z arrays in *extra*.
    """
    if R <= 0 or r <= 0:
        raise ValueError(f"Radii must be positive, got R={R}, r={r}.")
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return DescriptiveResult(
        name="torus_parametric",
        value=None,
        extra={"x": x, "y": y, "z": z, "R": R, "r": r},
    )


def cheatsheet() -> str:
    return "torus_parametric(R, r, u, v) -> parametric (x,y,z) on torus"
