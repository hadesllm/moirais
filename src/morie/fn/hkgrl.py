# morie.fn -- function file (hadesllm/morie)
"""Hawking radiation temperature model. 'History is about to be rewritten.' -- Hawkgirl"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_G = 6.67430e-11
_C = 2.99792458e8
_HBAR = 1.054571817e-34
_KB = 1.380649e-23


def hawking_temperature(
    mass_solar: float = 1.0,
    *,
    n_points: int = 50,
    mass_range: tuple[float, float] | None = None,
) -> DescriptiveResult:
    """Compute Hawking radiation temperature for a Schwarzschild black hole.

    T = hbar * c^3 / (8 * pi * G * M * k_B)

    Parameters
    ----------
    mass_solar : float
        Black hole mass in solar masses.
    n_points : int
        Number of points in the mass-temperature curve.
    mass_range : tuple or None
        (min, max) solar masses for the curve. Default (0.01, 100).

    Returns
    -------
    DescriptiveResult
        ``value`` = Hawking temperature in Kelvin.
    """
    M_SUN = 1.989e30
    if mass_solar <= 0:
        raise ValueError("mass must be positive")
    M = mass_solar * M_SUN
    T = _HBAR * _C**3 / (8 * np.pi * _G * M * _KB)
    if mass_range is None:
        mass_range = (0.01, 100.0)
    masses = np.geomspace(mass_range[0], mass_range[1], n_points)
    temps = _HBAR * _C**3 / (8 * np.pi * _G * masses * M_SUN * _KB)
    r_s = 2 * _G * M / _C**2
    evap_time = 5120 * np.pi * _G**2 * M**3 / (_HBAR * _C**4)
    return DescriptiveResult(
        name="Hawking radiation temperature",
        value=float(T),
        extra={
            "mass_solar": mass_solar,
            "mass_kg": M,
            "schwarzschild_radius_m": float(r_s),
            "evaporation_time_s": float(evap_time),
            "evaporation_time_yr": float(evap_time / 3.154e7),
            "curve_masses": masses.tolist(),
            "curve_temps": temps.tolist(),
        },
    )


hkgrl = hawking_temperature


def cheatsheet() -> str:
    return "hawking_temperature({}) -> Hawking radiation temperature model. 'History is about to be"
