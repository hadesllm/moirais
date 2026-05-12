# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian nonparametric contraction rate analysis. 'Contract, beliefs must.'"""

from __future__ import annotations

import numpy as np


def bayesian_contraction_rate(
    n: int,
    prior_entropy: float | None = None,
    prior_concentration: float = 1.0,
    dimension: int = 1,
    smoothness: float = 2.0,
) -> dict:
    r"""
    Compute Bayesian nonparametric contraction rate via Ghosal-van der Vaart theory.

    General contraction theorem requires:
    1. Metric entropy of model class
    2. Prior concentration around true parameter
    3. Test quality

    .. math::

        \epsilon_n^2 \geq \max(H(\Theta_n), \delta_n^2)

    where :math:`H(\Theta_n)` is metric entropy and :math:`\delta_n` is testing rate.

    :param n: Sample size.
    :param prior_entropy: Log prior probability of tests. If None, uses default.
    :param prior_concentration: DP concentration parameter.
    :param dimension: Problem dimension.
    :param smoothness: Holder smoothness.
    :return: Dictionary with contraction rate and analysis.
    """
    # Metric entropy for Holder class (standard result)
    # H(Theta_eps, rho) ~ (1/eps)^d
    # Taking logs: log H ~ d log(1/eps)

    # For Holder smoothness beta on [0,1]^d:
    # eps_n ~ n^{-beta/(2*beta + d)}

    exponent = -smoothness / (2.0 * smoothness + dimension)
    metric_rate = (n ** exponent) / np.sqrt(np.log(n))

    if prior_entropy is None:
        # Default: DP prior
        prior_entropy_rate = np.sqrt(np.log(prior_concentration) / n)
    else:
        prior_entropy_rate = prior_entropy

    # Testing rate (likelihood ratio test)
    test_rate = np.sqrt(dimension * np.log(n) / n)

    # Overall contraction (largest rate governs)
    contraction = max(metric_rate, prior_entropy_rate, test_rate)

    # Conditions for contraction
    cond_metric = metric_rate
    cond_prior = prior_entropy_rate
    cond_test = test_rate

    return {
        "contraction_rate": float(contraction),
        "metric_entropy_rate": float(cond_metric),
        "prior_concentration_rate": float(cond_prior),
        "testing_rate": float(cond_test),
        "exponent": exponent,
        "dimension": dimension,
        "smoothness": smoothness,
        "prior_concentration": prior_concentration,
        "n": n,
        "applies_when": {
            "metric_dominated": cond_metric > max(cond_prior, cond_test),
            "prior_dominated": cond_prior > max(cond_metric, cond_test),
            "test_dominated": cond_test > max(cond_metric, cond_prior),
        },
    }


bcntr = bayesian_contraction_rate


def cheatsheet() -> str:
    return "bayesian_contraction_rate(n, dimension=1, smoothness=2.0) -> BNP contraction"
