# moirais.fn — function file (hadesllm/moirais)
"""Hohmann transfer orbit computation. 'Ain't no thing like me, except me.' -- Rocket"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hohmann_transfer(
    r1: float,
    r2: float,
    *,
    mu: float = 3.986004418e14,
) -> DescriptiveResult:
    """Compute a Hohmann transfer orbit between two circular orbits.

    Parameters
    ----------
    r1 : float
        Radius of inner orbit in metres.
    r2 : float
        Radius of outer orbit in metres.
    mu : float
        Gravitational parameter (default: Earth, m^3/s^2).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``delta_v1``, ``delta_v2``, ``delta_v_total``,
        ``transfer_time`` (seconds), ``semi_major_transfer``.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbit radii must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")

    a_t = (r1 + r2) / 2
    v1_circ = np.sqrt(mu / r1)
    v2_circ = np.sqrt(mu / r2)
    v1_transfer = np.sqrt(mu * (2 / r1 - 1 / a_t))
    v2_transfer = np.sqrt(mu * (2 / r2 - 1 / a_t))
    dv1 = abs(v1_transfer - v1_circ)
    dv2 = abs(v2_circ - v2_transfer)
    t_transfer = np.pi * np.sqrt(a_t**3 / mu)

    return DescriptiveResult(
        name="hohmann_transfer",
        value={
            "delta_v1": float(dv1),
            "delta_v2": float(dv2),
            "delta_v_total": float(dv1 + dv2),
            "transfer_time": float(t_transfer),
            "semi_major_transfer": float(a_t),
        },
        extra={"r1": r1, "r2": r2, "mu": mu},
    )


rcoon = hohmann_transfer


def cheatsheet() -> str:
    return "hohmann_transfer({}) -> Hohmann transfer orbit computation. 'Ain't no thing like me,"
