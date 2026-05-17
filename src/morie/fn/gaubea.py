# morie.fn -- function file (hadesllm/morie)
"""Compute Gaussian beam parameters along the propagation axis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gaussian_beam(
    wavelength: float,
    w0: float,
    z: np.ndarray | list[float],
) -> DescriptiveResult:
    r"""Compute Gaussian beam parameters along the propagation axis.

    Calculates beam radius :math:`w(z)`, radius of curvature :math:`R(z)`,
    and Gouy phase :math:`\\psi(z)` for a TEM00 mode.

    Parameters
    ----------
    wavelength : float
        Wavelength in metres.
    w0 : float
        Beam waist radius in metres.
    z : array-like
        Propagation distances from the waist in metres.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``w`` (beam radius), ``R`` (curvature),
        ``gouy`` (phase), ``z_rayleigh``, ``divergence``.
    """
    if wavelength <= 0 or w0 <= 0:
        raise ValueError("wavelength and w0 must be positive")

    z_arr = np.asarray(z, dtype=float)
    z_R = np.pi * w0**2 / wavelength

    w = w0 * np.sqrt(1 + (z_arr / z_R) ** 2)

    R = np.where(
        np.abs(z_arr) < 1e-30,
        np.inf,
        z_arr * (1 + (z_R / z_arr) ** 2),
    )

    gouy = np.arctan(z_arr / z_R)
    divergence = wavelength / (np.pi * w0)

    return DescriptiveResult(
        name="gaussian_beam",
        value={
            "w": w,
            "R": R,
            "gouy": gouy,
            "z_rayleigh": float(z_R),
            "divergence": float(divergence),
        },
        extra={"wavelength": wavelength, "w0": w0},
    )


gaubea = gaussian_beam


def cheatsheet() -> str:
    return 'gaussian_beam({}) -> Gaussian beam optics.'
