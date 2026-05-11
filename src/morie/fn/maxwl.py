# morie.fn — function file (hadesllm/morie)
"""Maxwell equations (EM field tensor)."""

__all__ = ["maxwl"]

import numpy as np


def maxwl(
    E: np.ndarray,
    B: np.ndarray,
    c: float = 299792458.0,
) -> dict:
    """
    Construct the electromagnetic field tensor (Faraday tensor) and
    its dual, and compute Lorentz invariants.

    .. math::

        F^{\\mu\\nu} = \\begin{pmatrix}
            0 & -E_x/c & -E_y/c & -E_z/c \\\\
            E_x/c & 0 & -B_z & B_y \\\\
            E_y/c & B_z & 0 & -B_x \\\\
            E_z/c & -B_y & B_x & 0
        \\end{pmatrix}

    Lorentz invariants:

    .. math::

        I_1 = \\frac{1}{2} F_{\\mu\\nu} F^{\\mu\\nu} = B^2 - E^2/c^2

    .. math::

        I_2 = \\frac{1}{4} F_{\\mu\\nu} \\tilde{F}^{\\mu\\nu}
            = \\vec{E} \\cdot \\vec{B} / c

    Parameters
    ----------
    E : np.ndarray
        Electric field 3-vector (V/m).
    B : np.ndarray
        Magnetic field 3-vector (T).
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: field_tensor (4x4), dual_tensor (4x4),
        invariant_1, invariant_2, energy_density, poynting_vector.
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    if E.shape != (3,) or B.shape != (3,):
        raise ValueError("E and B must be 3-vectors.")

    F = np.zeros((4, 4))
    F[0, 1] = -E[0] / c; F[0, 2] = -E[1] / c; F[0, 3] = -E[2] / c
    F[1, 0] =  E[0] / c; F[1, 2] = -B[2];     F[1, 3] =  B[1]
    F[2, 0] =  E[1] / c; F[2, 1] =  B[2];     F[2, 3] = -B[0]
    F[3, 0] =  E[2] / c; F[3, 1] = -B[1];     F[3, 2] =  B[0]

    Fd = np.zeros((4, 4))
    Fd[0, 1] = -B[0]; Fd[0, 2] = -B[1]; Fd[0, 3] = -B[2]
    Fd[1, 0] =  B[0]; Fd[1, 2] =  E[2] / c; Fd[1, 3] = -E[1] / c
    Fd[2, 0] =  B[1]; Fd[2, 1] = -E[2] / c; Fd[2, 3] =  E[0] / c
    Fd[3, 0] =  B[2]; Fd[3, 1] =  E[1] / c; Fd[3, 2] = -E[0] / c

    eta = np.diag([1.0, -1.0, -1.0, -1.0])
    F_lower = eta @ F @ eta
    I1 = 0.5 * np.sum(F_lower * F)
    I2 = np.dot(E, B) / c

    eps0 = 8.8541878128e-12
    mu0 = 1.2566370614e-6
    u = 0.5 * (eps0 * np.dot(E, E) + np.dot(B, B) / mu0)
    S = np.cross(E, B) / mu0

    return {
        "field_tensor": F,
        "dual_tensor": Fd,
        "invariant_1": float(I1),
        "invariant_2": float(I2),
        "energy_density": float(u),
        "poynting_vector": S,
    }
