# moirais.fn — function file (hadesllm/moirais)
"""Mastering others is strength; mastering yourself is true power. — Lao Tzu"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kepler_orbit(
    a: float,
    e: float,
    n_points: int = 360,
) -> DescriptiveResult:
    """
    Compute a Keplerian orbit in 2D Cartesian coordinates.

    .. math::

        r(\\theta) = \\frac{a(1 - e^2)}{1 + e\\cos\\theta}

    :param a: Semi-major axis. Must be > 0.
    :param e: Eccentricity. Must be in [0, 1).
    :param n_points: Number of points around the orbit. Default 360.
    :return: DescriptiveResult with orbital parameters and coordinates.
    :raises ValueError: If a <= 0 or e not in [0, 1).

    References
    ----------
    Murray, C. D., & Dermott, S. F. (1999). *Solar System Dynamics*.
    Cambridge University Press.
    """
    if a <= 0:
        raise ValueError(f"Semi-major axis must be > 0, got {a}.")
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must be in [0, 1), got {e}.")

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    b = a * np.sqrt(1 - e**2)
    perihelion = a * (1 - e)
    aphelion = a * (1 + e)

    return DescriptiveResult(
        name="Keplerian Orbit",
        value=float(a),
        extra={
            "x": x,
            "y": y,
            "r": r,
            "theta": theta,
            "semi_minor": float(b),
            "perihelion": float(perihelion),
            "aphelion": float(aphelion),
            "eccentricity": float(e),
        },
    )


short = kepler_orbit


def cheatsheet() -> str:
    return "kepler_orbit({}) -> Keplerian orbit generator. 'I have a bad feeling about this."
