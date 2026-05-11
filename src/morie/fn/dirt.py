# morie.fn — function file (hadesllm/morie)
"""Dynamic IRT with random walk priors."""

from __future__ import annotations

from ._containers import DescriptiveResult


def dynamic_irt_model(
    votes,
    time_periods,
    n_samples: int = 500,
    burn_in: int = 100,
    seed: int = 42,
) -> DescriptiveResult:
    """Dynamic IRT with time-varying ideal points.

    :param votes: (n_leg x n_votes) binary vote matrix.
    :param time_periods: (n_votes,) integer time index.
    :param n_samples: MCMC samples.
    :param burn_in: Burn-in samples.
    :param seed: Random seed.
    :return: DescriptiveResult with ideal trajectories.

    .. epigraph:: "Kamehameha!" -- Goku, Dragon Ball Z
    """
    from morie._spatial_voting import dynamic_irt as _fn

    result = _fn(votes, time_periods, n_samples=n_samples, burn_in=burn_in, seed=seed)
    return DescriptiveResult(
        name="dynamic_irt_model",
        value=result["tau_mean"],
        extra=result,
    )


dirt = dynamic_irt_model


def cheatsheet() -> str:
    return "dynamic_irt_model({}) -> Dynamic IRT with random walk priors."
