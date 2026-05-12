# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian Information Criterion. 'Size matters not.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bayesian_info_criterion(
    loglik: float,
    n: int,
    k: int,
) -> DescriptiveResult:
    """Bayesian Information Criterion (Schwarz, 1978).

    BIC = -2 * loglik + k * ln(n)

    :param loglik: Maximised log-likelihood.
    :param n: Number of observations.
    :param k: Number of estimated parameters.
    :return: DescriptiveResult with BIC value.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    bic_val = -2.0 * loglik + k * np.log(n)
    return DescriptiveResult(
        name="bic",
        value=float(bic_val),
        extra={"loglik": loglik, "n": n, "k": k},
    )


bic = bayesian_info_criterion


def cheatsheet() -> str:
    return "bayesian_info_criterion({}) -> Bayesian Information Criterion. 'Size matters not.'"
