# morie.fn — function file (hadesllm/morie)
"""Dark energy equation of state. 'I am the end of all things.' -- Darkseid"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dark_energy_eos(
    z: np.ndarray | list[float] | None = None,
    *,
    w0: float = -1.0,
    wa: float = 0.0,
    omega_m: float = 0.3,
    omega_de: float = 0.7,
    h0: float = 70.0,
    n_points: int = 100,
) -> DescriptiveResult:
    """CPL dark energy equation of state and Hubble parameter as a function of redshift.

    Implements the Chevallier-Polarski-Linder (CPL) parameterization:
    w(z) = w0 + wa * z / (1 + z)

    The Hubble parameter H(z) is derived from the Friedmann equation in a
    flat (Omega_m + Omega_DE = 1) universe.

    Parameters
    ----------
    z : array or None
        Redshift values.  If None, uses linspace(0, 3, n_points).
    w0, wa : float
        CPL equation of state parameters. w0=-1, wa=0 is Lambda-CDM.
    omega_m, omega_de : float
        Present-day matter and dark energy density fractions.
    h0 : float
        Hubble constant in km/s/Mpc.
    n_points : int
        Grid points if *z* is None.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'z'``, ``'w_z'``, ``'H_z'``.
    """
    if z is None:
        z = np.linspace(0, 3, n_points)
    z = np.asarray(z, dtype=float)
    if np.any(z < 0):
        raise ValueError("Redshift must be non-negative")
    w_z = w0 + wa * z / (1 + z)
    de_factor = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    H_z = h0 * np.sqrt(omega_m * (1 + z) ** 3 + omega_de * de_factor)
    d_L = np.zeros_like(z)
    for i, zi in enumerate(z):
        if zi > 0:
            zz = np.linspace(0, zi, 200)
            ww = w0 + wa * zz / (1 + zz)
            de_f = (1 + zz) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * zz / (1 + zz))
            Hz = h0 * np.sqrt(omega_m * (1 + zz) ** 3 + omega_de * de_f)
            d_L[i] = (1 + zi) * np.trapezoid(1.0 / Hz, zz) * 2.998e5
    return DescriptiveResult(
        name="Dark energy equation of state (CPL)",
        value={"z": z.tolist(), "w_z": w_z.tolist(), "H_z": H_z.tolist()},
        extra={
            "w0": w0,
            "wa": wa,
            "omega_m": omega_m,
            "omega_de": omega_de,
            "h0": h0,
            "is_lambda_cdm": (w0 == -1.0 and wa == 0.0),
            "H0": h0,
            "H_z3": float(H_z[-1]) if len(H_z) > 0 else None,
            "luminosity_distance_Mpc": d_L.tolist(),
        },
    )


darks = dark_energy_eos


def cheatsheet() -> str:
    return "dark_energy_eos({}) -> Dark energy equation of state. 'I am the end of all things.'"
