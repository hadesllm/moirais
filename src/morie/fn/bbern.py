# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian Bernoulli model (Beta-Binomial conjugate)."""

from __future__ import annotations

__all__ = ["bayesian_bernoulli", "bbern"]

from typing import Any

from scipy import stats


def bayesian_bernoulli(
    successes: int,
    trials: int,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Beta-Binomial conjugate analysis for a Bernoulli probability.

    Prior: p ~ Beta(prior_a, prior_b)
    Likelihood: k | p ~ Binomial(n, p)

    Posterior: p | k, n ~ Beta(a + k, b + n - k)

    Parameters
    ----------
    successes : int
        Number of successes (k).
    trials : int
        Number of trials (n).
    prior_a : float
        Beta prior first shape parameter (alpha).
    prior_b : float
        Beta prior second shape parameter (beta).
    prob : float
        Credible interval probability.

    Returns
    -------
    dict
        posterior_mean, posterior_mode, posterior_var, ci_lower,
        ci_upper, posterior_a, posterior_b

    Raises
    ------
    ValueError
        If successes > trials or parameters invalid.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 2.
    """
    if trials < 0:
        raise ValueError("trials must be non-negative.")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials].")
    if prior_a <= 0 or prior_b <= 0:
        raise ValueError("prior_a and prior_b must be positive.")

    post_a = prior_a + successes
    post_b = prior_b + trials - successes

    post_mean = post_a / (post_a + post_b)
    post_var = (post_a * post_b) / ((post_a + post_b) ** 2 * (post_a + post_b + 1))

    if post_a > 1 and post_b > 1:
        post_mode = (post_a - 1) / (post_a + post_b - 2)
    else:
        post_mode = float("nan")

    ci_lo = float(stats.beta.ppf((1 - prob) / 2, post_a, post_b))
    ci_hi = float(stats.beta.ppf(1 - (1 - prob) / 2, post_a, post_b))

    return {
        "posterior_mean": float(post_mean),
        "posterior_mode": float(post_mode),
        "posterior_var": float(post_var),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "posterior_a": float(post_a),
        "posterior_b": float(post_b),
    }


bbern = bayesian_bernoulli


def cheatsheet() -> str:
    return "bayesian_bernoulli(successes, trials) -> Beta-Binomial conjugate analysis."
