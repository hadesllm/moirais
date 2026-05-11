# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian IRT log-likelihood evaluation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_irt_likelihood(votes, x, alpha, beta) -> DescriptiveResult:
    """Evaluate Bayesian IRT log-likelihood for a given parameter set.

    :param votes: Binary vote matrix (legislators x roll calls).
    :param x: Ideal point vector (legislators).
    :param alpha: Difficulty parameters (roll calls).
    :param beta: Discrimination parameters (roll calls).
    :return: DescriptiveResult with log-likelihood in ``extra``.

    .. epigraph:: "Those who break the rules are scum, but those who abandon their friends
       are worse than scum." -- Kakashi, Naruto
    """
    from morie._spatial_voting import bayesian_irt_likelihood as _fn

    result = _fn(votes, x, alpha, beta)
    return DescriptiveResult(
        name="bayesian_irt_likelihood",
        value=result["loglik"],
        extra=result,
    )


birtl = bayesian_irt_likelihood


def cheatsheet() -> str:
    return "bayesian_irt_likelihood({}) -> Bayesian IRT log-likelihood evaluation."
