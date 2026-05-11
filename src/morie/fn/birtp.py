# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian IRT posterior summarization."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_irt_posterior(chain, standardize: bool = True) -> DescriptiveResult:
    """Summarize posterior draws from a Bayesian IRT model.

    :param chain: MCMC chain dict with ideal points and item parameters.
    :param standardize: Whether to standardize ideal points to mean-zero.
    :return: DescriptiveResult with posterior summaries in ``extra``.

    .. epigraph:: "I am the hope of the universe." -- Goku, Dragon Ball Z
    """
    from morie._spatial_voting import bayesian_irt_posterior as _fn

    result = _fn(chain, standardize=standardize)
    return DescriptiveResult(
        name="bayesian_irt_posterior",
        value=result["n_samples"],
        extra=result,
    )


birtp = bayesian_irt_posterior


def cheatsheet() -> str:
    return "bayesian_irt_posterior({}) -> Bayesian IRT posterior summarization."
