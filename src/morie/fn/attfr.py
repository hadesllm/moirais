# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Population attributable fraction."""

from __future__ import annotations

from ._containers import ESRes


def attributable_fraction(
    RR: float,
    prevalence_exposure: float,
    *,
    alpha: float = 0.05,
    se_RR: float | None = None,
) -> ESRes:
    """
    Compute the population attributable fraction (PAF).

    .. math::

        PAF = \\frac{p_e (RR - 1)}{p_e (RR - 1) + 1}

    Parameters
    ----------
    RR : float
        Relative risk (or rate ratio).
    prevalence_exposure : float
        Prevalence of exposure in the population (0-1).
    alpha : float
        Significance level for CI.
    se_RR : float, optional
        Standard error of RR (for CI via delta method).

    Returns
    -------
    ESRes

    References
    ----------
    Levin, M. L. (1953). The occurrence of lung cancer in man.
    *Acta Unio Int Contra Cancrum*, 9(3), 531-541.
    """
    if RR <= 0:
        raise ValueError("RR must be positive.")
    if not (0 <= prevalence_exposure <= 1):
        raise ValueError("prevalence_exposure must be in [0, 1].")

    pe = prevalence_exposure
    paf = pe * (RR - 1) / (pe * (RR - 1) + 1)

    ci_lo, ci_hi = None, None
    se = None
    if se_RR is not None and se_RR > 0:
        from scipy.stats import norm

        dpaf_drr = pe / (pe * (RR - 1) + 1) ** 2
        se = abs(dpaf_drr) * se_RR
        z = norm.ppf(1 - alpha / 2)
        ci_lo = paf - z * se
        ci_hi = paf + z * se

    return ESRes(
        measure="PAF",
        estimate=float(paf),
        ci_lower=float(ci_lo) if ci_lo is not None else None,
        ci_upper=float(ci_hi) if ci_hi is not None else None,
        se=float(se) if se is not None else None,
        extra={"RR": RR, "prevalence_exposure": pe},
    )


attfr = attributable_fraction


def cheatsheet() -> str:
    return "attributable_fraction({}) -> Population attributable fraction."
