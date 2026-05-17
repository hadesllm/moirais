# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Aldrich-McKelvey scaling for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def aldrich_mckelvey_scaling(
    Z,
    n_dims: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Aldrich-McKelvey scaling of perceptual data.

    :param Z: Respondent x stimulus placement matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: DescriptiveResult with scaled positions in ``extra``.

    .. epigraph:: Give me a place to stand and I will move the earth. -- Archimedes
    """
    from morie._spatial_voting import aldrich_mckelvey as _fn

    result = _fn(Z, n_dims=n_dims, max_iter=max_iter, tol=tol)
    return DescriptiveResult(
        name="aldrich_mckelvey_scaling",
        value=result["iterations"],
        extra=result,
    )


amscl = aldrich_mckelvey_scaling


def cheatsheet() -> str:
    return "aldrich_mckelvey_scaling({}) -> Aldrich-McKelvey scaling for spatial voting."
