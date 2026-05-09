# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Anchoring vignettes for DIF correction."""

from __future__ import annotations

from ._containers import DescriptiveResult


def anchoring_vignettes_fn(
    Y,
    V,
    n_categories: int = 5,
) -> DescriptiveResult:
    """Anchoring vignettes for cross-group comparability.

    :param Y: Self-placement ratings.
    :param V: Respondent x vignette rating matrix.
    :param n_categories: Number of ordered categories.
    :return: DescriptiveResult with corrected scores.

    .. epigraph:: "I am Groot." -- Groot, Marvel
    """
    from moirais._spatial_voting import anchoring_vignettes as _fn

    result = _fn(Y, V, n_categories=n_categories)
    return DescriptiveResult(
        name="anchoring_vignettes",
        value=result["n_respondents"],
        extra=result,
    )


avign = anchoring_vignettes_fn


def cheatsheet() -> str:
    return "anchoring_vignettes_fn({}) -> Anchoring vignettes for DIF correction."
