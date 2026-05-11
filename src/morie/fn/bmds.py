# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian multidimensional scaling."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_mds_fn(
    D,
    n_dims: int = 2,
    n_samples: int = 1000,
    burn_in: int = 200,
    sigma_init: float = 1.0,
) -> DescriptiveResult:
    """Bayesian MDS with posterior coordinate uncertainty.

    :param D: Square distance matrix.
    :param n_dims: Number of dimensions.
    :param n_samples: Number of posterior samples.
    :param burn_in: Burn-in samples to discard.
    :param sigma_init: Initial noise scale.
    :return: DescriptiveResult with posterior coordinates in ``extra``.

    .. epigraph:: "I have been and always shall be your friend." -- Spock, Star Trek
    """
    from morie._spatial_voting import bayesian_mds as _fn

    result = _fn(D, n_dims=n_dims, n_samples=n_samples, burn_in=burn_in, sigma_init=sigma_init)
    return DescriptiveResult(
        name="bayesian_mds",
        value=result["sigma_mean"],
        extra=result,
    )


bmds = bayesian_mds_fn


def cheatsheet() -> str:
    return "bayesian_mds_fn({}) -> Bayesian multidimensional scaling."
