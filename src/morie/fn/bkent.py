# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bekenstein-Hawking entropy."""

__all__ = ["bkent"]

import numpy as np


def bkent(
    M: float = None,
    A: float = None,
    G: float = 6.67430e-11,
    c: float = 299792458.0,
    hbar: float = 1.0545718e-34,
    kB: float = 1.380649e-23,
) -> dict:
    r"""
    Compute Bekenstein-Hawking black hole entropy.

    .. math::

        S_{BH} = \\frac{k_B c^3 A}{4 G \\hbar}
               = \\frac{4 \\pi G M^2 k_B}{\\hbar c}

    Parameters
    ----------
    M : float, optional
        Black hole mass (kg). Provide M or A.
    A : float, optional
        Horizon area (m^2). If None, computed from M.
    G : float
        Gravitational constant.
    c : float
        Speed of light.
    hbar : float
        Reduced Planck constant.
    kB : float
        Boltzmann constant.

    Returns
    -------
    dict
        Keys: entropy_J_per_K, entropy_bits, horizon_area_m2,
        r_schwarzschild, temperature_K.
    """
    if M is not None:
        if M <= 0:
            raise ValueError("Mass must be > 0.")
        rs = 2.0 * G * M / c ** 2
        A = 4.0 * np.pi * rs ** 2
    elif A is not None:
        if A <= 0:
            raise ValueError("Area must be > 0.")
        rs = np.sqrt(A / (4.0 * np.pi))
        M = rs * c ** 2 / (2.0 * G)
    else:
        raise ValueError("Provide either M or A.")

    l_p2 = G * hbar / c ** 3
    S = kB * A / (4.0 * l_p2)
    S_bits = A / (4.0 * l_p2 * np.log(2.0))

    T_H = hbar * c ** 3 / (8.0 * np.pi * G * M * kB)

    return {
        "entropy_J_per_K": float(S),
        "entropy_bits": float(S_bits),
        "horizon_area_m2": float(A),
        "r_schwarzschild": float(rs),
        "temperature_K": float(T_H),
    }
