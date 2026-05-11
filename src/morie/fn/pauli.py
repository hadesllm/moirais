# morie.fn — function file (hadesllm/morie)
"""Pauli matrices and spin operators."""

__all__ = ["pauli"]

import numpy as np
from ._richresult import RichResult


def pauli(
    operator: str = "all",
    theta: float = 0.0,
    phi: float = 0.0,
) -> dict:
    """
    Compute Pauli matrices and spin-1/2 operators.

    .. math::

        \\sigma_x = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}, \\quad
        \\sigma_y = \\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}, \\quad
        \\sigma_z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}

    General spin operator along direction (theta, phi):

    .. math::

        \\hat{n} \\cdot \\vec{\\sigma} =
        \\sin\\theta \\cos\\phi \\, \\sigma_x
        + \\sin\\theta \\sin\\phi \\, \\sigma_y
        + \\cos\\theta \\, \\sigma_z

    Parameters
    ----------
    operator : str
        'x', 'y', 'z', 'plus', 'minus', 'n_dot_sigma', or 'all'.
    theta : float
        Polar angle for n_dot_sigma.
    phi : float
        Azimuthal angle for n_dot_sigma.

    Returns
    -------
    dict
        Keys depend on operator. 'all' returns sigma_x, sigma_y, sigma_z,
        sigma_plus, sigma_minus, identity.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sp = np.array([[0, 1], [0, 0]], dtype=complex)
    sm = np.array([[0, 0], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    if operator == "all":
        return {
            "sigma_x": sx,
            "sigma_y": sy,
            "sigma_z": sz,
            "sigma_plus": sp,
            "sigma_minus": sm,
            "identity": I2,
        }
    elif operator == "x":
        return RichResult(payload={"operator": sx})
    elif operator == "y":
        return RichResult(payload={"operator": sy})
    elif operator == "z":
        return RichResult(payload={"operator": sz})
    elif operator == "plus":
        return RichResult(payload={"operator": sp})
    elif operator == "minus":
        return RichResult(payload={"operator": sm})
    elif operator == "n_dot_sigma":
        n_sig = (
            np.sin(theta) * np.cos(phi) * sx
            + np.sin(theta) * np.sin(phi) * sy
            + np.cos(theta) * sz
        )
        eigenvalues = np.linalg.eigvalsh(n_sig)
        return {
            "operator": n_sig,
            "eigenvalues": np.sort(np.real(eigenvalues)),
        }
    else:
        raise ValueError(f"Unknown operator: {operator}")
