# moirais.fn — function file (hadesllm/moirais)
"""Ordered Optimal Classification for ordinal scales."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ordered_oc(
    Y,
    n_dims: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Ordered Optimal Classification for ordinal issue scales.

    :param Y: Respondent x item ordinal response matrix.
    :param n_dims: Number of latent dimensions.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: DescriptiveResult with ideal points and cutpoints.

    .. epigraph:: "The spice must flow." -- Duke Leto, Dune
    """
    from moirais._spatial_voting import ordered_optimal_classification as _fn

    result = _fn(Y, n_dims=n_dims, max_iter=max_iter, tol=tol)
    return DescriptiveResult(
        name="ordered_oc",
        value=result["correct_class"],
        extra=result,
    )


orooc = ordered_oc


def cheatsheet() -> str:
    return "ordered_oc({}) -> Ordered Optimal Classification for ordinal scales."
