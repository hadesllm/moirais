"""W-NOMINATE estimation (utility + log-likelihood)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def wnominate_estimate(
    votes,
    x,
    z_yea,
    z_nay,
    beta: float = 15.0,
    w=None,
) -> DescriptiveResult:
    """Full W-NOMINATE estimation combining utility and log-likelihood.

    :param votes: Binary vote matrix (legislators x roll calls).
    :param x: Ideal point matrix (legislators x dims).
    :param z_yea: Yea outcome locations (roll calls x dims).
    :param z_nay: Nay outcome locations (roll calls x dims).
    :param beta: Signal-to-noise ratio.
    :param w: Optional dimension salience weights.
    :return: DescriptiveResult with GMP and log-likelihood in ``extra``.

    .. epigraph:: "Whoever wins this war becomes justice." -- Donquixote Doflamingo, One Piece
    """
    from morie._spatial_voting import nominate_loglik as _fn

    result = _fn(votes, x, z_yea, z_nay, beta=beta, w=w)
    gmp = result.get("GMP", result.get("correct_class", 0.0))
    return DescriptiveResult(
        name="wnominate_estimate",
        value=gmp,
        extra=result,
    )


wnom = wnominate_estimate


def cheatsheet() -> str:
    return "wnominate_estimate({}) -> W-NOMINATE estimation (utility + log-likelihood)."
