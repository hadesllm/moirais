"""Power variogram model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def power_variogram(h, c0, c1, alpha=1.5):
    """Evaluate the power variogram model (unbounded).

    gamma(h) = c0 + c1 * h^alpha, 0 < alpha < 2.

    .. epigraph:: "A hunter must hunt." -- Bloodborne

    Parameters
    ----------
    h : array_like
        Lag distances.
    c0 : float
        Nugget.
    c1 : float
        Scale coefficient.
    alpha : float
        Power (must be in (0, 2)).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    if not (0 < alpha < 2):
        raise ValueError("alpha must be in (0, 2)")

    gamma = np.where(h > 0, c0 + c1 * h**alpha, 0.0)

    return DescriptiveResult(
        name="power_variogram",
        value=float(c1),
        extra={
            "gamma": gamma.tolist() if hasattr(gamma, "tolist") else [float(gamma)],
            "model": "power",
            "nugget": c0,
            "c1": c1,
            "alpha": alpha,
        },
    )


sgpow = power_variogram


def cheatsheet() -> str:
    return "power_variogram({}) -> Power variogram model."
