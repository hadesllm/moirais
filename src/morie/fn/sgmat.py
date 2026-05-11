"""Matern variogram model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def matern_variogram(h, nugget, sill, range_param, nu=1.5):
    """Evaluate the Matern variogram model.

    .. epigraph:: "Grant us eyes." -- Micolash, Bloodborne

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
    nu : float
        Smoothness parameter (0.5=exponential, inf=Gaussian).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.special import gamma as gamma_fn
    from scipy.special import kv

    h = np.asarray(h, dtype=np.float64)
    scaled = np.sqrt(2 * nu) * h / range_param

    with np.errstate(divide="ignore", invalid="ignore"):
        coef = (2 ** (1 - nu)) / gamma_fn(nu)
        cov = np.where(
            scaled > 0,
            coef * (scaled**nu) * kv(nu, scaled),
            1.0,
        )
    cov = np.clip(cov, 0, 1)
    gamma = np.where(h > 0, nugget + (sill - nugget) * (1.0 - cov), 0.0)

    return DescriptiveResult(
        name="matern_variogram",
        value=float(sill),
        extra={
            "gamma": gamma.tolist() if hasattr(gamma, "tolist") else [float(gamma)],
            "model": "matern",
            "nugget": nugget,
            "sill": sill,
            "range": range_param,
            "nu": nu,
        },
    )


sgmat = matern_variogram


def cheatsheet() -> str:
    return "matern_variogram({}) -> Matern variogram model."
