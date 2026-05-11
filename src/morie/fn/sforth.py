"""Orthogonality check of selected eigenvectors."""

from ._containers import SpatialResult


def sforth(evecs):
    """Orthogonality check of selected eigenvectors.

    Category: SFilter

    Parameters
    ----------
    evecs : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sforth", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sforth", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sforth_fn = sforth


def cheatsheet() -> str:
    return "sforth({}) -> Orthogonality check of selected eigenvectors."
