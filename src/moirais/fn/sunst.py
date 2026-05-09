"""Surface roughness metrics. 'Looking good is half the battle.' -- Sunstreaker"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def surface_roughness(
    profile: np.ndarray,
    *,
    dx: float = 1.0,
) -> DescriptiveResult:
    """Compute surface roughness parameters from a 1-D profile.

    Computes Ra (arithmetic average), Rq (RMS), Rz (max peak-to-valley),
    Rsk (skewness), and Rku (kurtosis) per ISO 4287.

    Parameters
    ----------
    profile : array-like
        1-D surface height profile.
    dx : float
        Sampling interval (for length calculation).

    Returns
    -------
    DescriptiveResult
        With ``value`` = Ra and ``extra`` containing all roughness params.
    """
    z = np.asarray(profile, dtype=float).ravel()
    if len(z) < 3:
        raise ValueError("Profile must have at least 3 points")

    z_centered = z - z.mean()
    ra = float(np.mean(np.abs(z_centered)))
    rq = float(np.sqrt(np.mean(z_centered**2)))
    rz = float(z.max() - z.min())
    rsk = float(np.mean(z_centered**3) / rq**3) if rq > 0 else 0.0
    rku = float(np.mean(z_centered**4) / rq**4) if rq > 0 else 0.0
    length = dx * (len(z) - 1)

    return DescriptiveResult(
        name="surface_roughness",
        value=ra,
        extra={"Ra": ra, "Rq": rq, "Rz": rz, "Rsk": rsk, "Rku": rku, "profile_length": length, "n_points": len(z)},
    )


sunst = surface_roughness


def cheatsheet() -> str:
    return "surface_roughness({}) -> Surface roughness metrics. 'Looking good is half the battle."
