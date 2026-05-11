# morie.fn — function file (hadesllm/morie)
"""Population Attributable Fraction (PAF)."""

from __future__ import annotations

from typing import Any


def population_attributable_fraction(
    prevalence_exposed: float,
    relative_risk: float,
    *,
    ci_rr: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """
    Compute the Population Attributable Fraction (PAF).

    .. math::

        \\text{PAF} = \\frac{p_e (RR - 1)}{1 + p_e (RR - 1)}

    where p_e = prevalence of exposure in the population and RR is the
    relative risk (or rate ratio) for the exposure-disease relationship.

    If CI for RR is provided, PAF CI is computed by substituting RR
    bounds into the formula (delta method approximation).

    :param prevalence_exposed: Proportion exposed in the population (0, 1).
    :param relative_risk: Relative risk (RR > 0).
    :param ci_rr: Optional tuple (RR_lower, RR_upper) for CI on RR.
    :return: Dictionary with paf, ci_lower, ci_upper.
    :raises ValueError: If inputs out of valid range.

    References
    ----------
    Levin, M. L. (1953). The occurrence of lung cancer in man. *Acta
    Unio Internationalis Contra Cancrum*, 9, 531--541.

    Rockhill, B., Newman, B., & Weinberg, C. (1998). Use and misuse of
    population attributable fractions. *AJPH*, 88(1), 15--19.
    """
    pe = prevalence_exposed
    rr = relative_risk
    if not (0 < pe < 1):
        raise ValueError("prevalence_exposed must be in (0, 1).")
    if rr <= 0:
        raise ValueError("relative_risk must be positive.")

    denom = 1.0 + pe * (rr - 1.0)
    paf = pe * (rr - 1.0) / denom

    ci_lo = None
    ci_hi = None
    if ci_rr is not None:
        rr_lo, rr_hi = ci_rr
        d_lo = 1.0 + pe * (rr_lo - 1.0)
        d_hi = 1.0 + pe * (rr_hi - 1.0)
        ci_lo = pe * (rr_lo - 1.0) / d_lo if d_lo != 0 else float("nan")
        ci_hi = pe * (rr_hi - 1.0) / d_hi if d_hi != 0 else float("nan")

    return {
        "paf": float(paf),
        "ci_lower": float(ci_lo) if ci_lo is not None else None,
        "ci_upper": float(ci_hi) if ci_hi is not None else None,
    }


paf = population_attributable_fraction


def cheatsheet() -> str:
    return "population_attributable_fraction({}) -> Population Attributable Fraction (PAF)."
