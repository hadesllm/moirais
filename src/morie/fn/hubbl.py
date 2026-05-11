# morie.fn — function file (hadesllm/morie)
"""Hubble parameter and expansion rate."""

__all__ = ["hubbl"]

import numpy as np
from scipy.integrate import quad


def hubbl(
    z: float,
    H0: float = 67.4,
    Omega_m: float = 0.315,
    Omega_r: float = 9.0e-5,
    Omega_Lambda: float = None,
) -> dict:
    """
    Compute Hubble parameter H(z) and related quantities at redshift z.

    .. math::

        H(z) = H_0 \\sqrt{
            \\Omega_r (1+z)^4 + \\Omega_m (1+z)^3
            + \\Omega_k (1+z)^2 + \\Omega_\\Lambda
        }

    Parameters
    ----------
    z : float
        Redshift (>= 0).
    H0 : float
        Hubble constant (km/s/Mpc).
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_Lambda : float, optional
        Dark energy. Default: flat universe closure.

    Returns
    -------
    dict
        Keys: H_z (km/s/Mpc), hubble_time_Gyr, lookback_time_Gyr, age_Gyr.
    """
    if z < 0:
        raise ValueError("Redshift z must be >= 0.")

    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m - Omega_r
    Omega_k = 1.0 - Omega_m - Omega_r - Omega_Lambda

    def E(zz):
        return np.sqrt(
            Omega_r * (1 + zz) ** 4
            + Omega_m * (1 + zz) ** 3
            + Omega_k * (1 + zz) ** 2
            + Omega_Lambda
        )

    H_z = H0 * E(z)

    km_per_Mpc = 3.0856775814913673e19
    s_per_Gyr = 3.1557e16
    t_H = km_per_Mpc / (H0 * s_per_Gyr)

    def integrand_lookback(zz):
        return 1.0 / ((1 + zz) * E(zz))

    lookback, _ = quad(integrand_lookback, 0, z)
    lookback_Gyr = lookback * t_H

    age_total, _ = quad(integrand_lookback, 0, np.inf)
    age_Gyr = age_total * t_H

    return {
        "H_z": float(H_z),
        "hubble_time_Gyr": t_H,
        "lookback_time_Gyr": float(lookback_Gyr),
        "age_Gyr": float(age_Gyr),
    }
