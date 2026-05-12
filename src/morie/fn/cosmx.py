# morie.fn -- function file (hadesllm/morie)
"""Comoving distance and lookback time."""

__all__ = ["cosmx"]

import numpy as np
from scipy.integrate import quad


def cosmx(
    z: float,
    H0: float = 67.4,
    Omega_m: float = 0.315,
    Omega_Lambda: float = None,
) -> dict:
    r"""
    Compute comoving distance, luminosity distance, angular diameter distance,
    and lookback time for a flat Lambda-CDM cosmology.

    .. math::

        d_C(z) = \\frac{c}{H_0} \\int_0^z \\frac{dz'}{E(z')}

    Parameters
    ----------
    z : float
        Redshift (>= 0).
    H0 : float
        Hubble constant (km/s/Mpc).
    Omega_m : float
        Matter density parameter.
    Omega_Lambda : float, optional
        Dark energy density. Default: 1 - Omega_m (flat).

    Returns
    -------
    dict
        Keys: comoving_distance_Mpc, luminosity_distance_Mpc,
        angular_diameter_distance_Mpc, lookback_time_Gyr.
    """
    if z < 0:
        raise ValueError("z must be >= 0.")

    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m

    c_km_s = 299792.458
    d_H = c_km_s / H0

    def E(zz):
        return np.sqrt(Omega_m * (1 + zz) ** 3 + Omega_Lambda)

    dc_val, _ = quad(lambda zz: 1.0 / E(zz), 0, z)
    dc = d_H * dc_val

    dl = dc * (1.0 + z)
    da = dc / (1.0 + z)

    km_per_Mpc = 3.0856775814913673e19
    s_per_Gyr = 3.1557e16
    t_H = km_per_Mpc / (H0 * s_per_Gyr)

    tl_val, _ = quad(lambda zz: 1.0 / ((1 + zz) * E(zz)), 0, z)
    tl = t_H * tl_val

    return {
        "comoving_distance_Mpc": float(dc),
        "luminosity_distance_Mpc": float(dl),
        "angular_diameter_distance_Mpc": float(da),
        "lookback_time_Gyr": float(tl),
    }
