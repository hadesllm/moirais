# morie.fn -- function file (hadesllm/morie)
"""NOMINATE log-likelihood computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_loglikelihood(
    votes,
    x,
    z_yea,
    z_nay,
    beta: float = 15.0,
    w=None,
) -> DescriptiveResult:
    """Compute NOMINATE log-likelihood for a spatial voting model.

    :param votes: Binary vote matrix (legislators x roll calls).
    :param x: Ideal point matrix (legislators x dims).
    :param z_yea: Yea outcome locations (roll calls x dims).
    :param z_nay: Nay outcome locations (roll calls x dims).
    :param beta: Signal-to-noise ratio.
    :param w: Optional dimension salience weights.
    :return: DescriptiveResult with log-likelihood in ``extra``.

    .. epigraph:: "Yeah, I'm thinking I'm back." -- John Wick
    """
    from morie._spatial_voting import nominate_loglik as _fn

    result = _fn(votes, x, z_yea, z_nay, beta=beta, w=w)
    return DescriptiveResult(
        name="nominate_loglikelihood",
        value=result["loglik"],
        extra=result,
    )


nomll = nominate_loglikelihood


def cheatsheet() -> str:
    return "nominate_loglikelihood({}) -> NOMINATE log-likelihood computation."
