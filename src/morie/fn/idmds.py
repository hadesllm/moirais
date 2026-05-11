# morie.fn — function file (hadesllm/morie)
"""INDSCAL: Individual Differences MDS."""

from __future__ import annotations

from ._containers import DescriptiveResult


def indscal_mds(
    dissimilarities,
    n_dims: int = 2,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """INDSCAL weighted MDS (Carroll & Chang 1970).

    :param dissimilarities: List of dissimilarity matrices per individual.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum ALS iterations.
    :param tol: Convergence tolerance.
    :return: DescriptiveResult with group config and weights.

    .. epigraph:: "Live long and prosper." -- Spock, Star Trek
    """
    from morie._spatial_voting import indscal as _fn

    result = _fn(dissimilarities, n_dims=n_dims, max_iter=max_iter, tol=tol)
    return DescriptiveResult(
        name="indscal_mds",
        value=result["stress"],
        extra=result,
    )


idmds = indscal_mds


def cheatsheet() -> str:
    return "indscal_mds({}) -> INDSCAL: Individual Differences MDS."
