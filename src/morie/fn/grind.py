# morie.fn -- function file (hadesllm/morie)
"""Particle size distribution (Rosin-Rammler). 'Grind them down!' -- Grindor"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rosin_rammler(
    sizes: np.ndarray,
    cum_passing: np.ndarray,
) -> DescriptiveResult:
    """Fit a Rosin-Rammler (Weibull) distribution to particle size data.

    Model: R(d) = exp(-(d/d_star)^n)
    where R is cumulative fraction retained, d_star is characteristic size,
    n is the distribution modulus.

    Parameters
    ----------
    sizes : array-like
        Particle sizes (e.g., sieve apertures).
    cum_passing : array-like
        Cumulative fraction passing (0 to 1).

    Returns
    -------
    DescriptiveResult
        With ``value`` = dict(d_star, n) and ``extra`` containing
        R-squared and fitted values.
    """
    d = np.asarray(sizes, dtype=float).ravel()
    cp = np.asarray(cum_passing, dtype=float).ravel()
    if len(d) != len(cp):
        raise ValueError("sizes and cum_passing must have same length")
    if len(d) < 3:
        raise ValueError("Need at least 3 data points")

    mask = (cp > 0.001) & (cp < 0.999) & (d > 0)
    d_f = d[mask]
    cp_f = cp[mask]
    if len(d_f) < 2:
        raise ValueError("Not enough valid data points after filtering")

    retained = 1 - cp_f
    y = np.log(np.log(1 / retained))
    x = np.log(d_f)

    slope, intercept = np.polyfit(x, y, 1)
    n = slope
    d_star = np.exp(-intercept / n)

    fitted_retained = np.exp(-((d_f / d_star) ** n))
    fitted_passing = 1 - fitted_retained
    ss_res = np.sum((cp_f - fitted_passing) ** 2)
    ss_tot = np.sum((cp_f - cp_f.mean()) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-30)

    return DescriptiveResult(
        name="rosin_rammler",
        value={"d_star": float(d_star), "n": float(n)},
        extra={"r_squared": float(r_squared), "fitted_passing": fitted_passing, "n_points": len(d_f)},
    )


grind = rosin_rammler


def cheatsheet() -> str:
    return "rosin_rammler({}) -> Particle size distribution (Rosin-Rammler). 'Grind them down"
