# moirais.fn — function file (hadesllm/moirais)
"""Linear PM₂.₅ / PM₁₀ relative-risk model (ACS / Harvard Six Cities)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# Landmark per-10-µg/m³ all-cause mortality RR estimates:
#   Pope et al. 2002 ACS: RR = 1.04  (PM2.5, 95% CI 1.01-1.08)
#   Dockery 1993 Six Cities: RR = 1.13 (PM2.5, 95% CI 1.04-1.23)
#   Di et al. 2017 NEJM:  RR = 1.073 (PM2.5, <12 µg/m³ subset)
# Outcome-specific estimates from Pope/Krewski 2009 extended analyses.
# Values below are the pooled central estimates most commonly cited.
_RR_PER_10_UGM3: dict[tuple[str, str], tuple[float, tuple[float, float]]] = {
    # (pollutant, outcome) -> (RR, (95% CI lo, hi))
    ("pm25", "all_cause"):         (1.070, (1.04, 1.10)),
    ("pm25", "cardiopulmonary"):   (1.090, (1.03, 1.16)),
    ("pm25", "ischemic_heart"):    (1.180, (1.14, 1.23)),
    ("pm25", "lung_cancer"):       (1.080, (1.01, 1.16)),
    ("pm25", "stroke"):            (1.110, (1.05, 1.17)),
    ("pm10", "all_cause"):         (1.040, (1.02, 1.06)),
    ("pm10", "respiratory"):       (1.040, (1.01, 1.07)),
}


def pm_relative_risk(
    concentration_ugm3: float | np.ndarray,
    *,
    reference_ugm3: float = 0.0,
    pollutant: str = "pm25",
    outcome: str = "all_cause",
) -> DescriptiveResult:
    """Linear log-linear PM exposure relative-risk estimator.

    Computes RR = exp( β × (C − C_ref) / 10 ) where β = ln(RR_10),
    using published per-10-µg/m³ estimates from the ACS cohort (Pope
    2002/Krewski 2009) and Harvard Six Cities family of studies.

    .. math::

        RR(C) = \\exp\\!\\left( \\ln(RR_{10}) \\cdot
        \\frac{C - C_{\\mathrm{ref}}}{10} \\right)

    Parameters
    ----------
    concentration_ugm3 : float or array-like
        Target PM concentration, µg/m³.
    reference_ugm3 : float, default 0.0
        Counterfactual reference. Use 5 for WHO 2021 PM₂.₅ AQG
        benchmarking, or the cohort's background for cohort studies.
    pollutant : {"pm25", "pm10"}, default "pm25"
        Particulate size fraction.
    outcome : str, default "all_cause"
        Outcome name. See `_RR_PER_10_UGM3` keys for available pairs.

    Returns
    -------
    DescriptiveResult
        value = RR at the mean concentration (dimensionless).
        extra has per-observation RR, the base 10-µg RR used, 95% CI
        on RR at each concentration, and source citation.

    Raises
    ------
    KeyError
        If (pollutant, outcome) pair not in the published-estimate
        table. Error lists available keys.

    Examples
    --------
    Delhi background annual PM₂.₅ ≈ 100 µg/m³ vs. WHO AQG of 5:

    >>> r = pm_relative_risk(100, reference_ugm3=5, outcome="all_cause")
    >>> round(r.value, 2)
    1.9

    References
    ----------
    Pope, C. A., Burnett, R. T., Thun, M. J., et al. (2002). Lung
    cancer, cardiopulmonary mortality, and long-term exposure to fine
    particulate air pollution. JAMA, 287(9), 1132-1141.

    Krewski, D., Jerrett, M., Burnett, R. T., et al. (2009). Extended
    follow-up and spatial analysis of the American Cancer Society
    study linking particulate air pollution and mortality. HEI
    Research Report 140.

    Di, Q., Wang, Y., Zanobetti, A., et al. (2017). Air pollution and
    mortality in the Medicare population. NEJM, 376(26), 2513-2522.

    Notes
    -----
    Quote: "Without data, you're just another person with an opinion."
    — W. Edwards Deming.
    """
    key = (pollutant.lower().strip(), outcome.lower().strip())
    if key not in _RR_PER_10_UGM3:
        raise KeyError(
            f"No published estimate for pollutant={pollutant!r}, "
            f"outcome={outcome!r}. Available: {sorted(_RR_PER_10_UGM3)}"
        )

    rr10, (lo10, hi10) = _RR_PER_10_UGM3[key]

    C = np.atleast_1d(np.asarray(concentration_ugm3, dtype=float))
    delta = (C - reference_ugm3) / 10.0
    rr = np.exp(np.log(rr10) * delta)
    rr_lo = np.exp(np.log(lo10) * delta)
    rr_hi = np.exp(np.log(hi10) * delta)

    val = float(rr.mean()) if rr.size > 1 else float(rr.item())

    return DescriptiveResult(
        name="pm_relative_risk",
        value=val,
        extra={
            "rr": rr.tolist() if rr.size > 1 else float(rr.item()),
            "rr_95lo": rr_lo.tolist() if rr_lo.size > 1 else float(rr_lo.item()),
            "rr_95hi": rr_hi.tolist() if rr_hi.size > 1 else float(rr_hi.item()),
            "rr_per_10_ugm3": rr10,
            "pollutant": key[0],
            "outcome": key[1],
            "reference_ugm3": reference_ugm3,
            "source": "Pope 2002 / Krewski 2009 / Di 2017",
        },
    )


pmrr = pm_relative_risk


def cheatsheet() -> str:
    return "pmrr(C, pollutant='pm25', outcome='all_cause') -> log-linear RR."
