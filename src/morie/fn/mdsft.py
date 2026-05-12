# morie.fn -- function file (hadesllm/morie)
"""MDS goodness-of-fit statistics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def mds_fit_statistics(eigenvalues) -> DescriptiveResult:
    """Compute MDS fit statistics from eigenvalues.

    :param eigenvalues: Array of eigenvalues from MDS decomposition.
    :return: DescriptiveResult with cumulative fit in ``extra``.

    .. epigraph:: "Resistance is futile." -- Borg, Star Trek
    """
    from morie._spatial_voting import mds_fit_stats as _fn

    result = _fn(eigenvalues)
    cum = result.get("cumulative_fit", [])
    return DescriptiveResult(
        name="mds_fit_statistics",
        value=cum[0] if len(cum) > 0 else 0.0,
        extra=result,
    )


mdsft = mds_fit_statistics


def cheatsheet() -> str:
    return "mds_fit_statistics({}) -> MDS goodness-of-fit statistics."
