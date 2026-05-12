# morie.fn -- function file (hadesllm/morie)
"""Dirac equation (free particle)."""

__all__ = ["dirac"]

import numpy as np


def dirac(
    p: np.ndarray,
    m: float,
    spin: str = "up",
    particle: str = "particle",
    hbar: float = 1.0545718e-34,
    c: float = 299792458.0,
) -> dict:
    r"""
    Compute the free-particle Dirac spinor and energy.

    The Dirac equation:

    .. math::

        (i \\hbar \\gamma^\\mu \\partial_\\mu - mc) \\psi = 0

    For a plane wave with 3-momentum p, the energy-momentum relation:

    .. math::

        E = \\sqrt{|\\mathbf{p}|^2 c^2 + m^2 c^4}

    Parameters
    ----------
    p : np.ndarray
        3-momentum [px, py, pz] (kg*m/s).
    m : float
        Rest mass (kg).
    spin : str
        'up' or 'down'.
    particle : str
        'particle' or 'antiparticle'.
    hbar : float
        Reduced Planck constant.
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: energy, spinor (4-component), gamma_matrices (list of 4x4).
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (3,):
        raise ValueError("p must be a 3-vector.")
    if m < 0:
        raise ValueError("Mass must be >= 0.")

    E = np.sqrt(np.sum(p ** 2) * c ** 2 + (m * c ** 2) ** 2)
    mc2 = m * c ** 2

    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Z2 = np.zeros((2, 2), dtype=complex)

    gamma0 = np.block([[I2, Z2], [Z2, -I2]])
    gamma1 = np.block([[Z2, sx], [-sx, Z2]])
    gamma2 = np.block([[Z2, sy], [-sy, Z2]])
    gamma3 = np.block([[Z2, sz], [-sz, Z2]])

    sigma_dot_p = p[0] * c * sx + p[1] * c * sy + p[2] * c * sz

    if spin == "up":
        chi = np.array([1.0, 0.0], dtype=complex)
    elif spin == "down":
        chi = np.array([0.0, 1.0], dtype=complex)
    else:
        raise ValueError("spin must be 'up' or 'down'.")

    if particle == "particle":
        norm = np.sqrt((E + mc2) / (2.0 * E + 1e-300))
        lower = sigma_dot_p @ chi / (E + mc2 + 1e-300)
        spinor = norm * np.concatenate([chi, lower])
    elif particle == "antiparticle":
        norm = np.sqrt((E + mc2) / (2.0 * E + 1e-300))
        upper = sigma_dot_p @ chi / (E + mc2 + 1e-300)
        spinor = norm * np.concatenate([upper, chi])
    else:
        raise ValueError("particle must be 'particle' or 'antiparticle'.")

    return {
        "energy": float(E),
        "spinor": spinor,
        "gamma_matrices": [gamma0, gamma1, gamma2, gamma3],
    }
