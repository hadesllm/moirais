"""Clifford torus in S^3 subset R^4."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def clifford_torus(n_points: int = 500) -> DescriptiveResult:
    r"""Generate points on the Clifford torus in S^3 subset R^4.

    The Clifford torus is the set:
    :math:`(x_1, x_2, x_3, x_4) = \\frac{1}{\\sqrt{2}}(\\cos u, \\sin u, \\cos v, \\sin v)`

    It is a flat torus embedded in S^3, dividing it into two solid tori.

    :param n_points: Number of sample points.
    :return: DescriptiveResult with coordinates in R^4.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")
    n_side = max(2, int(np.sqrt(n_points)))
    u = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    x1 = inv_sqrt2 * np.cos(uu)
    x2 = inv_sqrt2 * np.sin(uu)
    x3 = inv_sqrt2 * np.cos(vv)
    x4 = inv_sqrt2 * np.sin(vv)
    return DescriptiveResult(
        name="clifford_torus",
        value=float(n_side**2),
        extra={
            "x1": x1.ravel(),
            "x2": x2.ravel(),
            "x3": x3.ravel(),
            "x4": x4.ravel(),
            "n_actual": n_side**2,
            "gaussian_curvature": 0.0,
        },
    )


def cheatsheet() -> str:
    return "clifford_torus(n_points) -> Clifford torus in S^3 subset R^4"
