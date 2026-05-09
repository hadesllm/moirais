"""Torus Gaussian and mean curvature."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_curvature(
    R: float = 3.0,
    r: float = 1.0,
    u: float | np.ndarray = 0.0,
    v: float | np.ndarray = 0.0,
) -> DescriptiveResult:
    """Gaussian and mean curvature of a torus at parameter (u, v).

    Gaussian curvature:
    :math:`K = \\frac{\\cos v}{r(R + r\\cos v)}`

    Mean curvature:
    :math:`H = \\frac{R + 2r\\cos v}{2r(R + r\\cos v)}`

    :param R: Major radius.
    :param r: Minor radius.
    :param u: Angle around the torus (not used in curvature, kept for API).
    :param v: Angle around the tube (radians).
    :return: DescriptiveResult with gaussian_K and mean_H in *extra*.
    """
    if R <= 0 or r <= 0:
        raise ValueError(f"Radii must be positive, got R={R}, r={r}.")
    v = np.asarray(v, dtype=float)
    denom = r * (R + r * np.cos(v))
    K = np.cos(v) / denom
    H = (R + 2.0 * r * np.cos(v)) / (2.0 * denom)
    return DescriptiveResult(
        name="torus_curvature",
        value=float(np.mean(K)) if np.ndim(K) > 0 else float(K),
        extra={"gaussian_K": K, "mean_H": H, "R": R, "r": r},
    )


def cheatsheet() -> str:
    return "torus_curvature(R, r, u, v) -> Gaussian & mean curvature"
