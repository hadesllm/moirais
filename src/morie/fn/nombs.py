# morie.fn — function file (hadesllm/morie)
"""Parametric bootstrap for NOMINATE standard errors."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_bootstrap_se(
    votes,
    ideal_points,
    normal_vectors_arr,
    cutpoints,
    n_boot: int = 100,
    seed: int = 42,
) -> DescriptiveResult:
    """Parametric bootstrap SE estimation (Lewis & Poole 2004).

    :param votes: (n_leg x n_votes) original vote matrix.
    :param ideal_points: (n_leg x dims) estimated ideal points.
    :param normal_vectors_arr: (n_votes x dims) normal vectors.
    :param cutpoints: (n_votes,) cutpoints.
    :param n_boot: Number of bootstrap replications.
    :param seed: Random seed.
    :return: DescriptiveResult with bootstrap SEs.

    .. epigraph:: "One Piece does exist!" -- Whitebeard, One Piece
    """
    from morie._spatial_voting import nominate_bootstrap as _fn

    result = _fn(votes, ideal_points, normal_vectors_arr, cutpoints, n_boot=n_boot, seed=seed)
    return DescriptiveResult(
        name="nominate_bootstrap_se",
        value=result["n_boot"],
        extra=result,
    )


nombs = nominate_bootstrap_se


def cheatsheet() -> str:
    return "nominate_bootstrap_se({}) -> Parametric bootstrap for NOMINATE standard errors."
