# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian metric unfolding."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_unfolding(
    D,
    n_dims: int = 2,
    n_samples: int = 1000,
    burn_in: int = 200,
) -> DescriptiveResult:
    """Bayesian metric unfolding via MCMC.

    :param D: Rectangular distance matrix (respondents x stimuli).
    :param n_dims: Number of dimensions.
    :param n_samples: Number of posterior samples.
    :param burn_in: Burn-in samples to discard.
    :return: DescriptiveResult with posterior samples in ``extra``.

    .. epigraph:: "A man's dream will never die!" -- Blackbeard, One Piece
    """
    from morie._spatial_voting import bayesian_unfolding as _fn

    result = _fn(D, n_dims=n_dims, n_samples=n_samples, burn_in=burn_in)
    return DescriptiveResult(
        name="bayesian_unfolding",
        value=result["n_samples"],
        extra=result,
    )


bunfl = bayesian_unfolding


def cheatsheet() -> str:
    return "bayesian_unfolding({}) -> Bayesian metric unfolding."
