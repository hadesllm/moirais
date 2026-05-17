# morie.fn -- function file (hadesllm/morie)
"""Optimal classification for roll-call voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def optimal_classification(
    votes,
    n_dims: int = 1,
    max_iter: int = 500,
    n_restarts: int = 10,
) -> DescriptiveResult:
    """Poole's Optimal Classification (OC) estimator.

    :param votes: Binary vote matrix (legislators x roll calls).
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations per restart.
    :param n_restarts: Number of random restarts.
    :return: DescriptiveResult with PRE and ideal points in ``extra``.

    .. epigraph:: You have power over your mind, not outside events. -- Marcus Aurelius
    """
    from morie._spatial_voting import optimal_classification as _fn

    result = _fn(votes, n_dims=n_dims, max_iter=max_iter, n_restarts=n_restarts)
    return DescriptiveResult(
        name="optimal_classification",
        value=result["PRE"],
        extra=result,
    )


oocls = optimal_classification


def cheatsheet() -> str:
    return "optimal_classification({}) -> Optimal classification for roll-call voting."
