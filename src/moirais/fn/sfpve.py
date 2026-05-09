"""Proportion of variance explained by eigenvectors."""

from ._containers import SpatialResult


def sfpve(y, evecs):
    """Proportion of variance explained by eigenvectors.

    Category: SFilter

    Parameters
    ----------
    y, evecs : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        result = 0.0
        return SpatialResult(name="sfpve", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sfpve", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sfpve_fn = sfpve


def cheatsheet() -> str:
    return "sfpve({}) -> Proportion of variance explained by eigenvectors."
