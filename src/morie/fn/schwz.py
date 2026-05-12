# morie.fn -- function file (hadesllm/morie)
"""Schwarzschild metric (black hole geometry)."""

__all__ = ["schwz"]

import numpy as np


def schwz(
    r: float,
    M: float,
    theta: float = np.pi / 2,
    G: float = 6.67430e-11,
    c: float = 299792458.0,
) -> dict:
    r"""
    Compute the Schwarzschild metric tensor at radius r.

    The line element in Schwarzschild coordinates (t, r, theta, phi):

    .. math::

        ds^2 = -\\left(1 - \\frac{r_s}{r}\\right) c^2 dt^2
               + \\left(1 - \\frac{r_s}{r}\\right)^{-1} dr^2
               + r^2 d\\Omega^2

    where :math:`r_s = 2GM/c^2`.

    Parameters
    ----------
    r : float
        Radial coordinate (must be > r_s).
    M : float
        Mass of the central body (kg).
    theta : float
        Polar angle (default pi/2, equatorial plane).
    G : float
        Gravitational constant.
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: metric (4x4), r_schwarzschild, f (metric function 1-rs/r).
    """
    if M <= 0:
        raise ValueError("Mass must be positive.")
    rs = 2.0 * G * M / (c ** 2)
    if r <= rs:
        raise ValueError(f"r must be > Schwarzschild radius {rs:.6e} m.")

    f = 1.0 - rs / r
    g = np.zeros((4, 4))
    g[0, 0] = -f * c ** 2
    g[1, 1] = 1.0 / f
    g[2, 2] = r ** 2
    g[3, 3] = r ** 2 * np.sin(theta) ** 2

    return {
        "metric": g,
        "r_schwarzschild": rs,
        "f": f,
    }
