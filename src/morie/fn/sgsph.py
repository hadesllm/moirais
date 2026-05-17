"""Spherical variogram model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def spherical_variogram(h, nugget, sill, range_param):
    """Evaluate the spherical variogram model.

    gamma(h) = nugget + (sill - nugget) * [1.5*(h/a) - 0.5*(h/a)^3] for h <= a.

    .. epigraph:: The Analytical Engine weaves algebraic patterns. -- Ada Lovelace

    Parameters
    ----------
    h : array_like
        Lag distances.
    nugget : float
        Nugget effect (C0).
    sill : float
        Total sill (C0 + C1).
    range_param : float
        Range parameter (a).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    gamma = np.where(
        h <= range_param,
        nugget + (sill - nugget) * (1.5 * h / range_param - 0.5 * (h / range_param) ** 3),
        sill,
    )
    gamma = np.where(h == 0, 0.0, gamma)

    return DescriptiveResult(
        name="spherical_variogram",
        value=float(sill),
        extra={
            "gamma": gamma.tolist() if hasattr(gamma, "tolist") else [float(gamma)],
            "model": "spherical",
            "nugget": nugget,
            "sill": sill,
            "range": range_param,
        },
    )


sgsph = spherical_variogram


def cheatsheet() -> str:
    return "spherical_variogram({}) -> Spherical variogram model."
