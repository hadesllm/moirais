# morie.fn — function file (hadesllm/morie)
"""Evidence Lower Bound (ELBO) computation."""

from __future__ import annotations

__all__ = ["compute_elbo", "elbo"]

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def compute_elbo(
    log_target: Callable[[np.ndarray], float],
    variational_mean: Union[list, np.ndarray],
    variational_std: Union[list, np.ndarray],
    *,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    r"""
    Estimate the Evidence Lower Bound (ELBO) for a mean-field
    Gaussian variational approximation.

    The ELBO decomposes as:

    .. math::

        \\text{ELBO} = \\mathbb{E}_q[\\log p(\\theta | y)]
                      - \\mathbb{E}_q[\\log q(\\theta)]
                     = \\mathbb{E}_q[\\log p(\\theta | y)]
                      + H[q]

    where H[q] is the entropy of the variational distribution.

    Parameters
    ----------
    log_target : callable
        Log-density of target (unnormalized OK).
    variational_mean : array-like
        Mean of the variational Gaussian (d,).
    variational_std : array-like
        Standard deviation of the variational Gaussian (d,).
    n_samples : int
        Number of Monte Carlo samples for the expectation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        elbo : float
        expected_log_target : float
        entropy : float
        elbo_se : float (standard error of the MC estimate)

    References
    ----------
    Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017).
    Variational inference: A review for statisticians. *JASA*,
    112(518), 859--877.
    Jordan, M. I., et al. (1999). Machine Learning, 37, 183--233.
    """
    mu = np.asarray(variational_mean, dtype=float)
    sigma = np.asarray(variational_std, dtype=float)
    d = len(mu)

    if len(sigma) != d:
        raise ValueError("variational_mean and variational_std must have same length.")
    if np.any(sigma <= 0):
        raise ValueError("variational_std must be strictly positive.")

    entropy = 0.5 * d * (1.0 + np.log(2 * np.pi)) + np.sum(np.log(sigma))

    rng = np.random.default_rng(seed)
    log_targets = np.empty(n_samples)

    for i in range(n_samples):
        theta = mu + sigma * rng.standard_normal(d)
        log_targets[i] = log_target(theta)

    expected_lp = float(np.mean(log_targets))
    elbo_val = expected_lp + entropy
    se = float(np.std(log_targets, ddof=1) / np.sqrt(n_samples))

    return {
        "elbo": elbo_val,
        "expected_log_target": expected_lp,
        "entropy": float(entropy),
        "elbo_se": se,
        "n_samples": n_samples,
    }


elbo = compute_elbo


def cheatsheet() -> str:
    return "compute_elbo(log_target, mean, std) -> Evidence Lower Bound (ELBO)."
