# morie.fn — function file (hadesllm/morie)
"""Clinton-Jackman-Rivers IRT model for roll-call data."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cjr_irt_model(
    votes,
    n_dims: int = 1,
    n_samples: int = 1000,
    burn_in: int = 200,
) -> DescriptiveResult:
    """Clinton-Jackman-Rivers Bayesian IRT for ideal point estimation.

    :param votes: Binary vote matrix (legislators x roll calls).
    :param n_dims: Number of latent dimensions.
    :param n_samples: Number of posterior samples.
    :param burn_in: Burn-in samples to discard.
    :return: DescriptiveResult with ideal point posteriors in ``extra``.

    .. epigraph:: "It's over 9000!" -- Vegeta, Dragon Ball Z
    """
    from morie._spatial_voting import cjr_irt as _fn

    result = _fn(votes, n_dims=n_dims, n_samples=n_samples, burn_in=burn_in)
    return DescriptiveResult(
        name="cjr_irt_model",
        value=result["n_samples"],
        extra=result,
    )


cjrit = cjr_irt_model


def cheatsheet() -> str:
    return "cjr_irt_model({}) -> Clinton-Jackman-Rivers IRT model for roll-call data."
