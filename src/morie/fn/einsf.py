# morie.fn -- function file (hadesllm/morie)
"""Einstein field equation tensor (G_uv = 8*pi*G/c^4 * T_uv)."""

__all__ = ["einsf"]

import numpy as np


def einsf(
    ricci_tensor: np.ndarray,
    scalar_curvature: float,
    metric: np.ndarray,
    stress_energy: np.ndarray = None,
    Lambda: float = 0.0,
    G: float = 6.67430e-11,
    c: float = 299792458.0,
) -> dict:
    r"""
    Compute the Einstein tensor and verify the field equations.

    .. math::

        G_{\\mu\\nu} = R_{\\mu\\nu} - \\frac{1}{2} R \\, g_{\\mu\\nu}

    .. math::

        G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu}
        = \\frac{8\\pi G}{c^4} T_{\\mu\\nu}

    Parameters
    ----------
    ricci_tensor : np.ndarray
        (n,n) Ricci tensor.
    scalar_curvature : float
        Ricci scalar R.
    metric : np.ndarray
        (n,n) metric tensor.
    stress_energy : np.ndarray, optional
        (n,n) stress-energy tensor. If provided, residual is computed.
    Lambda : float
        Cosmological constant (default 0).
    G : float
        Gravitational constant.
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: einstein_tensor (n,n), residual (n,n or None),
        kappa (coupling constant 8*pi*G/c^4).
    """
    n = metric.shape[0]
    G_tensor = ricci_tensor - 0.5 * scalar_curvature * metric

    kappa = 8.0 * np.pi * G / (c ** 4)

    residual = None
    if stress_energy is not None:
        lhs = G_tensor + Lambda * metric
        rhs = kappa * stress_energy
        residual = lhs - rhs

    return {
        "einstein_tensor": G_tensor,
        "residual": residual,
        "kappa": kappa,
    }
