"""Relative structured variability (RSV)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def relative_structured_variability(nugget, sill):
    """Compute relative structured variability RSV = (sill - nugget) / sill.

    RSV near 1 = strong spatial structure, near 0 = mostly noise.

    .. epigraph:: "Shaw!" -- The Knight, Hollow Knight

    Parameters
    ----------
    nugget : float
        Nugget effect.
    sill : float
        Total sill.

    Returns
    -------
    DescriptiveResult
    """
    rsv = (sill - nugget) / sill if sill > 0 else 0.0
    nsr = nugget / sill if sill > 0 else 1.0

    if rsv >= 0.75:
        strength = "strong"
    elif rsv >= 0.50:
        strength = "moderate"
    elif rsv >= 0.25:
        strength = "weak"
    else:
        strength = "very_weak"

    return DescriptiveResult(
        name="relative_structured_variability",
        value=float(rsv),
        extra={
            "RSV": float(rsv),
            "nugget_sill_ratio": float(nsr),
            "spatial_strength": strength,
            "nugget": float(nugget),
            "sill": float(sill),
        },
    )


sgrsv = relative_structured_variability


def cheatsheet() -> str:
    return "relative_structured_variability({}) -> Relative structured variability (RSV)."
