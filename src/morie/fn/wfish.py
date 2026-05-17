"""Wordfish Poisson IRT for text scaling."""

from __future__ import annotations

from ._containers import DescriptiveResult


def wordfish_scaling(
    dtm,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Wordfish Poisson IRT (Slapin & Proksch 2008).

    :param dtm: (n_docs x n_words) document-term count matrix.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: DescriptiveResult with document positions.

    .. epigraph:: We must know. We will know. -- David Hilbert
    """
    from morie._spatial_voting import wordfish_irt as _fn

    result = _fn(dtm, max_iter=max_iter, tol=tol)
    return DescriptiveResult(
        name="wordfish_scaling",
        value=result["log_lik"],
        extra=result,
    )


wfish = wordfish_scaling


def cheatsheet() -> str:
    return "wordfish_scaling({}) -> Wordfish Poisson IRT for text scaling."
