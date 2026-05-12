# morie.fn -- function file (hadesllm/morie)
"""Penrose diagram coordinates."""

__all__ = ["penrs"]

import numpy as np


def penrs(
    r: np.ndarray,
    t: np.ndarray,
    M: float = 1.0,
    geometry: str = "schwarzschild",
    G: float = 1.0,
    c: float = 1.0,
) -> dict:
    r"""
    Compute Penrose (conformal/compactified) diagram coordinates.

    For Schwarzschild geometry, uses Kruskal-Szekeres as intermediate:

    .. math::

        T = \\sqrt{\\frac{r}{r_s} - 1} \\, e^{r/(2r_s)} \\sinh\\left(\\frac{t}{2r_s}\\right)

    .. math::

        X = \\sqrt{\\frac{r}{r_s} - 1} \\, e^{r/(2r_s)} \\cosh\\left(\\frac{t}{2r_s}\\right)

    Then compactify: :math:`U = \\arctan(T - X)`, :math:`V = \\arctan(T + X)`.

    For Minkowski, tortoise then compactify:

    .. math::

        U = \\arctan(t - r), \\quad V = \\arctan(t + r)

    Parameters
    ----------
    r : np.ndarray
        Radial coordinates.
    t : np.ndarray
        Time coordinates (same shape as r).
    M : float
        Mass parameter.
    geometry : str
        'schwarzschild' or 'minkowski'.
    G : float
        Gravitational constant (natural units default).
    c : float
        Speed of light.

    Returns
    -------
    dict
        Keys: U (compactified), V (compactified), T_penrose, R_penrose.
    """
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)

    if geometry == "minkowski":
        u = t - r
        v = t + r
        U = np.arctan(u)
        V = np.arctan(v)
        T_p = (V + U) / 2.0
        R_p = (V - U) / 2.0
        return {
            "U": U,
            "V": V,
            "T_penrose": T_p,
            "R_penrose": R_p,
        }

    elif geometry == "schwarzschild":
        rs = 2.0 * G * M / c ** 2
        mask = r > rs
        if not np.all(mask):
            raise ValueError("All r must be > Schwarzschild radius for exterior region.")

        ratio = r / rs - 1.0
        exp_factor = np.exp(r / (2.0 * rs))
        sqrt_ratio = np.sqrt(ratio)

        T_ks = sqrt_ratio * exp_factor * np.sinh(c * t / (2.0 * rs))
        X_ks = sqrt_ratio * exp_factor * np.cosh(c * t / (2.0 * rs))

        u_null = T_ks - X_ks
        v_null = T_ks + X_ks

        U = np.arctan(u_null)
        V = np.arctan(v_null)
        T_p = (V + U) / 2.0
        R_p = (V - U) / 2.0

        return {
            "U": U,
            "V": V,
            "T_penrose": T_p,
            "R_penrose": R_p,
        }
    else:
        raise ValueError("geometry must be 'schwarzschild' or 'minkowski'.")
