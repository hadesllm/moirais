# morie.fn -- function file (hadesllm/morie)
"""Population attributable fraction (PAF)."""

from __future__ import annotations

from ._containers import ESRes


def population_attributable_fraction(
    rr: float,
    prevalence: float,
    rr_ci_lower: float | None = None,
    rr_ci_upper: float | None = None,
    confidence: float = 0.95,
) -> ESRes:
    """Population attributable fraction (PAF).

    .. math::

        PAF = \\frac{p_e (RR - 1)}{1 + p_e (RR - 1)}

    where :math:`p_e` is exposure prevalence and :math:`RR` is the
    relative risk.

    Parameters
    ----------
    rr : float
        Relative risk (or odds ratio for rare outcomes).
    prevalence : float
        Prevalence of exposure in the population.
    rr_ci_lower : float, optional
        Lower CI of RR (for PAF CI computation).
    rr_ci_upper : float, optional
        Upper CI of RR.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Levin, M. L. (1953). The occurrence of lung cancer in man.
    Acta Unio Internationalis Contra Cancrum, 9(3), 531-541.

    Rockhill, B. et al. (1998). Use and misuse of population
    attributable fractions. American Journal of Public Health,
    88(1), 15-19.
    """
    if rr <= 0:
        raise ValueError("rr must be positive")
    if not 0 <= prevalence <= 1:
        raise ValueError("prevalence must be in [0, 1]")

    paf = prevalence * (rr - 1) / (1 + prevalence * (rr - 1))

    ci_lo = ci_hi = None
    if rr_ci_lower is not None and rr_ci_upper is not None:
        ci_lo = prevalence * (rr_ci_lower - 1) / (1 + prevalence * (rr_ci_lower - 1))
        ci_hi = prevalence * (rr_ci_upper - 1) / (1 + prevalence * (rr_ci_upper - 1))

    return ESRes(
        measure="PAF",
        estimate=float(paf),
        ci_lower=float(ci_lo) if ci_lo is not None else None,
        ci_upper=float(ci_hi) if ci_hi is not None else None,
        extra={"rr": rr, "prevalence": prevalence},
    )


psafs = population_attributable_fraction


def cheatsheet() -> str:
    return "population_attributable_fraction({}) -> Population attributable fraction (PAF)."
