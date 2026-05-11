# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Beta-binomial conjugate Bayesian analysis."""

from __future__ import annotations

from scipy import stats

from ._containers import DescriptiveResult


def bayesian_binomial(
    successes: int,
    trials: int,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    prob: float = 0.95,
) -> DescriptiveResult:
    """
    Beta-binomial conjugate analysis for a proportion.

    Parameters
    ----------
    successes : int
        Number of successes.
    trials : int
        Number of trials.
    prior_a, prior_b : float
        Beta prior hyperparameters (default: uniform).
    prob : float
        Credible interval probability.

    Returns
    -------
    DescriptiveResult
        extra has 'posterior_mean', 'posterior_var', 'ci_lower',
        'ci_upper', 'prior', 'posterior'.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.
    CRC Press, Ch. 2.
    """
    if successes < 0 or trials <= 0 or successes > trials:
        raise ValueError("Invalid successes/trials.")

    post_a = prior_a + successes
    post_b = prior_b + trials - successes
    post_mean = post_a / (post_a + post_b)
    post_var = (post_a * post_b) / ((post_a + post_b) ** 2 * (post_a + post_b + 1))
    ci_lo = float(stats.beta.ppf((1 - prob) / 2, post_a, post_b))
    ci_hi = float(stats.beta.ppf(1 - (1 - prob) / 2, post_a, post_b))

    return DescriptiveResult(
        name="bayesian_binomial",
        value=float(post_mean),
        extra={
            "posterior_mean": float(post_mean),
            "posterior_var": float(post_var),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "prior": {"a": prior_a, "b": prior_b},
            "posterior": {"a": float(post_a), "b": float(post_b)},
        },
    )


bbin = bayesian_binomial


def cheatsheet() -> str:
    return "bayesian_binomial({}) -> Beta-binomial conjugate Bayesian analysis."
