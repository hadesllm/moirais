# moirais.fn — function file (hadesllm/moirais)
"""Maximum likelihood estimation for Gaussian."""

import numpy as np

from ._containers import ESRes
def mle_gaussian(x, **kwargs) -> ESRes:
    """
    Compute MLE of mean and variance for a Gaussian distribution.

    .. math::

        \\hat{\\mu} = \\frac{1}{n} \\sum_{i=1}^{n} x_i, \\quad
        \\hat{\\sigma}^2 = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\hat{\\mu})^2

    Note: MLE variance uses *n* (biased), not *n-1*.

    :param x: array-like of observations.
    :return: ESRes with MLE mean as estimate and variance in extra.

    References
    ----------
    Fisher RA (1922). On the mathematical foundations of theoretical
    statistics. Philosophical Transactions A, 222, 309-368.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 2:
        raise ValueError("Need at least 2 observations.")
    mu_hat = float(np.mean(x))
    var_hat = float(np.mean((x - mu_hat) ** 2))
    return ESRes(
        measure="mle_gaussian",
        estimate=mu_hat,
        n=len(x),
        extra={"mu": mu_hat, "sigma2": var_hat, "sigma": float(np.sqrt(var_hat))},
    )


mle = mle_gaussian


def cheatsheet() -> str:
    return "mle_gaussian({}) -> Maximum likelihood estimation for Gaussian."
