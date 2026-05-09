"""Shock wave Mach number. 'Feel the thunder!' -- Thundercracker"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mach_shock(
    velocity: float,
    *,
    speed_of_sound: float = 343.0,
    gamma: float = 1.4,
) -> DescriptiveResult:
    """Compute shock wave properties from Mach number.

    Uses normal shock relations from gas dynamics (Rankine-Hugoniot).

    Parameters
    ----------
    velocity : float
        Object velocity (m/s).
    speed_of_sound : float
        Speed of sound in medium (m/s). Default 343 m/s (air at 20C).
    gamma : float
        Heat capacity ratio (1.4 for air).

    Returns
    -------
    DescriptiveResult
        With ``value`` = Mach number and ``extra`` containing
        pressure ratio, temperature ratio, and shock cone angle.
    """
    if velocity < 0:
        raise ValueError("velocity must be non-negative")
    if speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be positive")

    M = velocity / speed_of_sound

    if M > 1:
        p_ratio = 1 + 2 * gamma / (gamma + 1) * (M**2 - 1)
        t_ratio = p_ratio * (2 + (gamma - 1) * M**2) / ((gamma + 1) * M**2)
        rho_ratio = (gamma + 1) * M**2 / (2 + (gamma - 1) * M**2)
        cone_angle = np.degrees(np.arcsin(1 / M))
        regime = "supersonic"
    elif M == 1:
        p_ratio = t_ratio = rho_ratio = 1.0
        cone_angle = 90.0
        regime = "sonic"
    else:
        p_ratio = t_ratio = rho_ratio = 1.0
        cone_angle = float("nan")
        regime = "subsonic"

    return DescriptiveResult(
        name="mach_shock",
        value=float(M),
        extra={
            "regime": regime,
            "pressure_ratio": float(p_ratio),
            "temperature_ratio": float(t_ratio),
            "density_ratio": float(rho_ratio),
            "cone_angle_deg": float(cone_angle),
            "velocity_ms": velocity,
            "gamma": gamma,
        },
    )


thndm = mach_shock


def cheatsheet() -> str:
    return "mach_shock({}) -> Shock wave Mach number. 'Feel the thunder!' -- Thundercracke"
