# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian binomial with beta conjugate prior."""

from __future__ import annotations

from typing import Any

from scipy import stats


def bayesian_binomial(
    successes: int,
    trials: int,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Beta-binomial conjugate Bayesian analysis.

    Prior: theta ~ Beta(a, b)
    Posterior: theta | data ~ Beta(a + s, b + n - s)

    :param successes: Number of successes.
    :param trials: Number of trials.
    :param prior_a: Beta prior alpha.
    :param prior_b: Beta prior beta.
    :param prob: Credible interval probability.
    :return: Dictionary with posterior parameters, mean, mode, HDI.

    References
    ----------
    Kruschke, J. K. (2015). *Doing Bayesian Data Analysis*, 2nd ed., Ch. 6.
    """
    s = int(successes)
    n = int(trials)
    post_a = prior_a + s
    post_b = prior_b + (n - s)

    post_mean = post_a / (post_a + post_b)
    post_var = (post_a * post_b) / ((post_a + post_b) ** 2 * (post_a + post_b + 1))
    post_mode = (post_a - 1) / (post_a + post_b - 2) if (post_a > 1 and post_b > 1) else float("nan")

    alpha_half = (1 - prob) / 2
    ci_lower = float(stats.beta.ppf(alpha_half, post_a, post_b))
    ci_upper = float(stats.beta.ppf(1 - alpha_half, post_a, post_b))

    from scipy.special import betaln

    log_ml = float(
        betaln(post_a, post_b) - betaln(prior_a, prior_b)
    )

    return {
        "posterior_a": float(post_a),
        "posterior_b": float(post_b),
        "posterior_mean": float(post_mean),
        "posterior_mode": float(post_mode),
        "posterior_var": float(post_var),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "log_marginal_likelihood": log_ml,
    }


bnoml = bayesian_binomial


def cheatsheet() -> str:
    return "bayesian_binomial({}) -> Bayesian binomial with beta conjugate prior."
