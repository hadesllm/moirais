# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian negative binomial (gamma conjugate)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_negbinom(
    counts: Union[list, np.ndarray],
    *,
    r: float = 5.0,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian inference for the negative binomial success probability p.

    Model: X_i ~ NB(r, p). Conjugate prior: p ~ Beta(a, b).
    Posterior: p | data ~ Beta(a + n*r, b + sum(x)).

    :param counts: Observed counts (k,).
    :param r: Known number of successes parameter.
    :param prior_a: Beta prior alpha.
    :param prior_b: Beta prior beta.
    :param prob: Credible interval probability.
    :return: Dictionary with posterior parameters and summaries.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.
    """
    c = np.asarray(counts, dtype=float).ravel()
    n = len(c)
    total = float(np.sum(c))

    post_a = prior_a + n * r
    post_b = prior_b + total

    post_mean = post_a / (post_a + post_b)
    post_var = (post_a * post_b) / ((post_a + post_b) ** 2 * (post_a + post_b + 1))

    alpha_half = (1 - prob) / 2
    ci_lower = float(stats.beta.ppf(alpha_half, post_a, post_b))
    ci_upper = float(stats.beta.ppf(1 - alpha_half, post_a, post_b))

    return {
        "posterior_a": float(post_a),
        "posterior_b": float(post_b),
        "posterior_mean": float(post_mean),
        "posterior_var": float(post_var),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r": r,
        "n": n,
    }


bnebi = bayesian_negbinom


def cheatsheet() -> str:
    return "bayesian_negbinom({}) -> Bayesian negative binomial (gamma conjugate)."
