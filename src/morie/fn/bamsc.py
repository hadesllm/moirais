# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian Aldrich-McKelvey scaling."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_am_scaling(
    Z,
    n_samples: int = 1000,
    burn_in: int = 200,
    prior_sd: float = 10.0,
) -> DescriptiveResult:
    """Bayesian Aldrich-McKelvey scaling via MCMC.

    :param Z: Respondent x stimulus placement matrix.
    :param n_samples: Number of posterior samples.
    :param burn_in: Burn-in samples to discard.
    :param prior_sd: Prior standard deviation on stimulus positions.
    :return: DescriptiveResult with posterior samples in ``extra``.

    .. epigraph:: "Arrakis teaches the attitude of the knife." -- Stilgar, Dune
    """
    from morie._spatial_voting import bayesian_am_scaling as _fn

    result = _fn(Z, n_samples=n_samples, burn_in=burn_in, prior_sd=prior_sd)
    return DescriptiveResult(
        name="bayesian_am_scaling",
        value=result["n_samples"],
        extra=result,
    )


bamsc = bayesian_am_scaling


def cheatsheet() -> str:
    return "bayesian_am_scaling({}) -> Bayesian Aldrich-McKelvey scaling."
