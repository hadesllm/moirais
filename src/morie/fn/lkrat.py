# morie.fn — function file (hadesllm/morie)
"""Likelihood ratio for Gaussian hypotheses."""

import numpy as np

from ._containers import ESRes

_QUOTE = "Statistics is the grammar of science. — Karl Pearson"


def likelihood_ratio(x, params0, params1, **kwargs) -> ESRes:
    """
    Compute likelihood ratio Λ = L(θ₁; x) / L(θ₀; x) for Gaussian.

    :param x: array-like of observations.
    :param params0: tuple (mu0, sigma0) for null hypothesis.
    :param params1: tuple (mu1, sigma1) for alternative hypothesis.
    :return: ESRes with likelihood ratio and log-likelihood ratio.
    :raises ValueError: If sigma values are non-positive.

    References
    ----------
    Neyman J, Pearson ES (1933). On the problem of the most efficient
    tests of statistical hypotheses. Philosophical Transactions A,
    231, 289-337.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mu0, sig0 = float(params0[0]), float(params0[1])
    mu1, sig1 = float(params1[0]), float(params1[1])
    if sig0 <= 0 or sig1 <= 0:
        raise ValueError("sigma must be > 0.")
    ll0 = float(np.sum(-0.5 * np.log(2 * np.pi * sig0**2) - 0.5 * ((x - mu0) / sig0) ** 2))
    ll1 = float(np.sum(-0.5 * np.log(2 * np.pi * sig1**2) - 0.5 * ((x - mu1) / sig1) ** 2))
    log_lr = ll1 - ll0
    lr = float(np.exp(np.clip(log_lr, -500, 500)))
    return ESRes(
        measure="likelihood_ratio",
        estimate=lr,
        n=len(x),
        extra={"log_lr": log_lr, "ll0": ll0, "ll1": ll1},
    )


lkrat = likelihood_ratio


def cheatsheet() -> str:
    return "likelihood_ratio({}) -> Likelihood ratio for Gaussian hypotheses."
