# morie.fn — function file (hadesllm/morie)
"""NOMINATE single vote probability."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_vote_prob_fn(
    x_i,
    z_yea_j,
    z_nay_j,
    beta: float = 15.0,
    w=None,
) -> DescriptiveResult:
    """Compute NOMINATE vote probability for one legislator-vote pair.

    :param x_i: Ideal point vector for legislator i.
    :param z_yea_j: Yea outcome location for roll call j.
    :param z_nay_j: Nay outcome location for roll call j.
    :param beta: Signal-to-noise ratio.
    :param w: Optional dimension salience weights.
    :return: DescriptiveResult with vote probability.

    .. epigraph:: "You think darkness is your ally." -- Bane, DC
    """
    from morie._spatial_voting import nominate_vote_prob as _fn

    result = _fn(x_i, z_yea_j, z_nay_j, beta=beta, w=w)
    return DescriptiveResult(
        name="nominate_vote_prob",
        value=result,
        extra={"prob": result},
    )


nomvt = nominate_vote_prob_fn


def cheatsheet() -> str:
    return "nominate_vote_prob_fn({}) -> NOMINATE single vote probability."
