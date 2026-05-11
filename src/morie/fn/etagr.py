# morie.fn — function file (hadesllm/morie)
"""Etiologic fraction (attributable fraction among exposed)."""

from __future__ import annotations

from ._containers import ESRes


def etiologic_fraction(
    rr: float,
    rr_ci_lower: float | None = None,
    rr_ci_upper: float | None = None,
) -> ESRes:
    """Etiologic fraction (attributable fraction among the exposed).

    .. math::

        EF = \\frac{RR - 1}{RR}

    Also known as the attributable risk percent (AR%).

    Parameters
    ----------
    rr : float
        Relative risk.
    rr_ci_lower : float, optional
        Lower CI of RR.
    rr_ci_upper : float, optional
        Upper CI of RR.

    Returns
    -------
    ESRes

    References
    ----------
    Miettinen, O. S. (1974). Proportion of disease caused or prevented
    by a given exposure, trait or intervention. American Journal of
    Epidemiology, 99(5), 325-332.
    """
    if rr <= 0:
        raise ValueError("rr must be positive")

    ef = (rr - 1) / rr

    ci_lo = ci_hi = None
    if rr_ci_lower is not None and rr_ci_upper is not None:
        if rr_ci_lower > 0 and rr_ci_upper > 0:
            ci_lo = (rr_ci_lower - 1) / rr_ci_lower
            ci_hi = (rr_ci_upper - 1) / rr_ci_upper

    return ESRes(
        measure="etiologic_fraction",
        estimate=float(ef),
        ci_lower=float(ci_lo) if ci_lo is not None else None,
        ci_upper=float(ci_hi) if ci_hi is not None else None,
        extra={"rr": rr},
    )


etagr = etiologic_fraction


def cheatsheet() -> str:
    return "etiologic_fraction({}) -> Etiologic fraction (AF among exposed)."
