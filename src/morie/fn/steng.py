"""Stress-energy tensor (perfect fluid)."""

__all__ = ["steng"]

import numpy as np


def steng(
    rho: float,
    p: float,
    u: np.ndarray,
    metric: np.ndarray,
    c: float = 299792458.0,
) -> dict:
    r"""
    Construct the stress-energy tensor for a perfect fluid.

    .. math::

        T^{\\mu\\nu} = \\left(\\rho + \\frac{p}{c^2}\\right) u^\\mu u^\\nu
                       + p \\, g^{\\mu\\nu}

    Parameters
    ----------
    rho : float
        Energy density (J/m^3).
    p : float
        Pressure (Pa).
    u : np.ndarray
        4-velocity (contravariant).
    metric : np.ndarray
        (4,4) metric tensor (covariant).
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: stress_energy (4x4, contravariant), trace (float).
    """
    u = np.asarray(u, dtype=float)
    metric = np.asarray(metric, dtype=float)
    if u.shape != (4,) or metric.shape != (4, 4):
        raise ValueError("u must be (4,), metric must be (4,4).")

    ginv = np.linalg.inv(metric)
    T = (rho + p / c ** 2) * np.outer(u, u) + p * ginv

    trace = 0.0
    for mu in range(4):
        for nu in range(4):
            trace += metric[mu, nu] * T[mu, nu]

    return {
        "stress_energy": T,
        "trace": float(trace),
    }
