# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Alpha-NOMINATE: Bayesian NOMINATE via MCMC."""

from __future__ import annotations

from ._containers import DescriptiveResult


def alpha_nominate_estimate(
    votes,
    n_dims: int = 2,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
) -> DescriptiveResult:
    """Alpha-NOMINATE Bayesian estimation (Carroll et al. 2013).

    :param votes: (n_leg x n_votes) binary vote matrix.
    :param n_dims: Number of latent dimensions.
    :param n_samples: MCMC samples after burn-in.
    :param burn_in: Burn-in samples.
    :param seed: Random seed.
    :return: DescriptiveResult with alpha and ideal points.

    .. epigraph:: The only true wisdom is in knowing you know nothing. -- Socrates
    """
    from morie._spatial_voting import alpha_nominate as _fn

    result = _fn(votes, n_dims=n_dims, n_samples=n_samples, burn_in=burn_in, seed=seed)
    return DescriptiveResult(
        name="alpha_nominate_estimate",
        value=result["alpha_mean"],
        extra=result,
    )


anom = alpha_nominate_estimate


def cheatsheet() -> str:
    return "alpha_nominate_estimate({}) -> Alpha-NOMINATE: Bayesian NOMINATE via MCMC."
