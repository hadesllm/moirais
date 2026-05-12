# morie.fn -- function file (hadesllm/morie)
"""MAP estimate with Gaussian prior."""

import numpy as np

from ._containers import ESRes
def map_estimate(
    x,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    **kwargs,
) -> ESRes:
    r"""
    Maximum a posteriori estimate of the mean with a Gaussian prior.

    .. math::

        \\hat{\\mu}_{MAP} = \\frac{\\frac{n \\bar{x}}{\\sigma^2_{MLE}}
        + \\frac{\\mu_0}{\\sigma_0^2}}{\\frac{n}{\\sigma^2_{MLE}}
        + \\frac{1}{\\sigma_0^2}}

    :param x: array-like of observations.
    :param prior_mu: Prior mean μ₀.
    :param prior_sigma: Prior standard deviation σ₀.
    :return: ESRes with MAP estimate.

    References
    ----------
    DeGroot MH (1970). Optimal Statistical Decisions.
    McGraw-Hill, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 1:
        raise ValueError("Need at least 1 observation.")
    if prior_sigma <= 0:
        raise ValueError("prior_sigma must be > 0.")
    xbar = float(np.mean(x))
    s2 = float(np.var(x, ddof=0)) if n > 1 else 1.0
    if s2 == 0:
        s2 = 1e-12
    tau2 = prior_sigma**2
    precision_data = n / s2
    precision_prior = 1.0 / tau2
    mu_map = (precision_data * xbar + precision_prior * prior_mu) / (precision_data + precision_prior)
    posterior_var = 1.0 / (precision_data + precision_prior)
    return ESRes(
        measure="map_estimate",
        estimate=float(mu_map),
        n=n,
        extra={
            "posterior_var": float(posterior_var),
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "mle_mean": xbar,
        },
    )


mapst = map_estimate


def cheatsheet() -> str:
    return "map_estimate({}) -> MAP estimate with Gaussian prior."
