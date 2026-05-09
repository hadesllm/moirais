# moirais.fn — function file (hadesllm/moirais)
"""Ising model partition function (2D, Onsager)."""

__all__ = ["ising"]

import numpy as np
from scipy.integrate import quad


def ising(
    T: float,
    J: float = 1.0,
    kB: float = 1.0,
    n_sites: int = None,
) -> dict:
    """
    Compute 2D Ising model thermodynamic quantities using Onsager's
    exact solution on an infinite square lattice.

    Critical temperature:

    .. math::

        k_B T_c = \\frac{2J}{\\ln(1 + \\sqrt{2})}

    Spontaneous magnetization (T < Tc):

    .. math::

        M = \\left[1 - \\sinh^{-4}(2\\beta J)\\right]^{1/8}

    Parameters
    ----------
    T : float
        Temperature (> 0).
    J : float
        Coupling constant (> 0 ferromagnetic).
    kB : float
        Boltzmann constant (default 1 for natural units).
    n_sites : int, optional
        If given, compute finite-lattice free energy per site.

    Returns
    -------
    dict
        Keys: T_c, beta_c, magnetization, free_energy_per_site,
        internal_energy_per_site, specific_heat_per_site, is_ordered.
    """
    if T <= 0:
        raise ValueError("Temperature must be > 0.")
    if J <= 0:
        raise ValueError("J must be > 0 (ferromagnetic).")

    Tc = 2.0 * J / (kB * np.log(1.0 + np.sqrt(2.0)))
    beta = 1.0 / (kB * T)
    beta_c = 1.0 / (kB * Tc)
    K = beta * J

    sinh_val = np.sinh(2.0 * K)
    if Tc > T:
        if sinh_val > 0:
            M = (1.0 - sinh_val ** (-4)) ** (1.0 / 8.0)
        else:
            M = 1.0
    else:
        M = 0.0

    kappa = 2.0 * np.sinh(2.0 * K) / np.cosh(2.0 * K) ** 2
    kappa = min(abs(kappa), 1.0 - 1e-15)

    def integrand(theta):
        return np.log(0.5 * (1.0 + np.sqrt(1.0 - kappa ** 2 * np.sin(theta) ** 2)))

    integral_val, _ = quad(integrand, 0, np.pi / 2)

    f_per_site = -(kB * T) * (
        np.log(2.0 * np.cosh(2.0 * K))
        + (2.0 / np.pi) * integral_val
    )

    coth_val = np.cosh(2.0 * K) / (np.sinh(2.0 * K) + 1e-300)
    K1_kappa = np.pi / 2
    if abs(kappa) > 1e-10:
        def K1_integrand(theta):
            return 1.0 / np.sqrt(1.0 - kappa ** 2 * np.sin(theta) ** 2)
        K1_kappa, _ = quad(K1_integrand, 0, np.pi / 2)

    u_per_site = -J * coth_val * (
        1.0 + (2.0 / np.pi) * (2.0 * np.tanh(2.0 * K) ** 2 - 1.0) * K1_kappa
    )

    return {
        "T_c": float(Tc),
        "beta_c": float(beta_c),
        "magnetization": float(M),
        "free_energy_per_site": float(f_per_site),
        "internal_energy_per_site": float(u_per_site),
        "is_ordered": bool(Tc > T),
    }
