"""SMACOF unfolding for rectangular distance data."""

from __future__ import annotations

from ._containers import DescriptiveResult


def smacof_unfolding(
    D,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """SMACOF-based metric unfolding.

    :param D: Rectangular distance matrix (respondents x stimuli).
    :param n_dims: Number of dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: DescriptiveResult with row/column coordinates in ``extra``.

    .. epigraph:: "Believe it!" -- Naruto Uzumaki, Naruto
    """
    from morie._spatial_voting import smacof_unfolding as _fn

    result = _fn(D, n_dims=n_dims, max_iter=max_iter, tol=tol)
    return DescriptiveResult(
        name="smacof_unfolding",
        value=result["stress"],
        extra=result,
    )


smcuf = smacof_unfolding


def cheatsheet() -> str:
    return "smacof_unfolding({}) -> SMACOF unfolding for rectangular distance data."
