"""Exponential variogram model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def exponential_variogram(h, nugget, sill, range_param):
    """Evaluate the exponential variogram model.

    gamma(h) = nugget + (sill - nugget) * (1 - exp(-h/a)).

    .. epigraph:: Statistics is the grammar of science. -- Karl Pearson

    Parameters
    ----------
    h : array_like
        Lag distances.
    nugget : float
        Nugget effect.
    sill : float
        Total sill.
    range_param : float
        Range parameter (practical range ~ 3*a).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    gamma = np.where(
        h > 0,
        nugget + (sill - nugget) * (1.0 - np.exp(-h / range_param)),
        0.0,
    )

    return DescriptiveResult(
        name="exponential_variogram",
        value=float(sill),
        extra={
            "gamma": gamma.tolist() if hasattr(gamma, "tolist") else [float(gamma)],
            "model": "exponential",
            "nugget": nugget,
            "sill": sill,
            "range": range_param,
            "practical_range": 3.0 * range_param,
        },
    )


sgexp = exponential_variogram


def cheatsheet() -> str:
    return "exponential_variogram({}) -> Exponential variogram model."
