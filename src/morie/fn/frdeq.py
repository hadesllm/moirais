# morie.fn — function file (hadesllm/morie)
"""Friedmann equations (cosmological expansion)."""

__all__ = ["frdeq"]

import numpy as np


def frdeq(
    H0: float = 67.4,
    Omega_m: float = 0.315,
    Omega_r: float = 9.0e-5,
    Omega_Lambda: float = None,
    k: int = 0,
    a_range: tuple = (0.01, 2.0),
    n_points: int = 500,
) -> dict:
    r"""
    Compute Friedmann equation quantities for FLRW cosmology.

    First Friedmann equation:

    .. math::

        H^2(a) = H_0^2 \\left[
            \\Omega_r a^{-4} + \\Omega_m a^{-3}
            + \\Omega_k a^{-2} + \\Omega_\\Lambda
        \\right]

    Parameters
    ----------
    H0 : float
        Hubble constant (km/s/Mpc). Default 67.4 (Planck 2018).
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_Lambda : float, optional
        Dark energy density. If None, computed for flat universe.
    k : int
        Curvature: 0 (flat), +1 (closed), -1 (open).
    a_range : tuple
        (a_min, a_max) scale factor range.
    n_points : int
        Number of evaluation points.

    Returns
    -------
    dict
        Keys: a (scale factors), H (Hubble parameter at each a),
        Omega_k, Omega_Lambda, deceleration_q (at each a).
    """
    if H0 <= 0:
        raise ValueError("H0 must be positive.")

    if Omega_Lambda is None:
        Omega_k = -float(k)
        Omega_Lambda = 1.0 - Omega_m - Omega_r - Omega_k
    else:
        Omega_k = 1.0 - Omega_m - Omega_r - Omega_Lambda

    a = np.linspace(a_range[0], a_range[1], n_points)

    H2 = H0 ** 2 * (
        Omega_r * a ** (-4)
        + Omega_m * a ** (-3)
        + Omega_k * a ** (-2)
        + Omega_Lambda
    )
    H2 = np.maximum(H2, 0.0)
    H = np.sqrt(H2)

    q = 0.5 * Omega_r * a ** (-4) + 0.5 * Omega_m * a ** (-3) - Omega_Lambda
    q = q / (H / H0) ** 2

    return {
        "a": a,
        "H": H,
        "Omega_k": Omega_k,
        "Omega_Lambda": Omega_Lambda,
        "deceleration_q": q,
    }
