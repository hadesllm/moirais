"""Gaussian variogram model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def gaussian_variogram(h, nugget, sill, range_param):
    """Evaluate the Gaussian variogram model.

    gamma(h) = nugget + (sill - nugget) * (1 - exp(-(h/a)^2)).

    .. epigraph:: The whole is greater than the sum of its parts. -- Aristotle

    Parameters
    ----------
    h : array_like
        Lag distances.
    nugget : float
        Nugget effect.
    sill : float
        Total sill.
    range_param : float
        Range parameter.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    gamma = np.where(
        h > 0,
        nugget + (sill - nugget) * (1.0 - np.exp(-((h / range_param) ** 2))),
        0.0,
    )

    return DescriptiveResult(
        name="gaussian_variogram",
        value=float(sill),
        extra={
            "gamma": gamma.tolist() if hasattr(gamma, "tolist") else [float(gamma)],
            "model": "gaussian",
            "nugget": nugget,
            "sill": sill,
            "range": range_param,
            "practical_range": range_param * np.sqrt(3.0),
        },
    )


sggau = gaussian_variogram


def cheatsheet() -> str:
    return "gaussian_variogram({}) -> Gaussian variogram model."
