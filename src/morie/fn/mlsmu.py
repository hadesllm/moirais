# morie.fn — function file (hadesllm/morie)
"""MLSMU6 metric unfolding."""

from __future__ import annotations

from ._containers import DescriptiveResult


def mlsmu6_unfolding(
    D,
    n_dims: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 5,
) -> DescriptiveResult:
    """MLSMU6 metric unfolding (Borg & Groenen).

    :param D: Rectangular distance matrix (respondents x stimuli).
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations per restart.
    :param tol: Convergence tolerance.
    :param n_restarts: Number of random restarts.
    :return: DescriptiveResult with coordinates and stress in ``extra``.

    .. epigraph:: "People's dreams never end!" -- Blackbeard, One Piece
    """
    from morie._spatial_voting import mlsmu6 as _fn

    result = _fn(D, n_dims=n_dims, max_iter=max_iter, tol=tol, n_restarts=n_restarts)
    return DescriptiveResult(
        name="mlsmu6_unfolding",
        value=result["stress"],
        extra=result,
    )


mlsmu = mlsmu6_unfolding


def cheatsheet() -> str:
    return "mlsmu6_unfolding({}) -> MLSMU6 metric unfolding."
