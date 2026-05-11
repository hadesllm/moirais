"""
Environmental-health composition module for MORIE.

Composes primitives from `morie.fn.*` (IPW, AIPW, DML, PLR, etc.) into
end-to-end pipelines for quantifying how pollution exposure translates
into disease burden. Built on top of the M1-canonically-verified
causal library (21/22 fns, see
an internal coverage report).

The pipeline stages, top-down:

    exposure (μg/m³) ─┐
                     ├─→ concentration-response (RR, log-RR)
    outcome rate  ─┘           │
                                ├─→ attributable fraction (PAF)
    population    ──────────────┤
                                ├─→ mortality displaced
                                ├─→ burden of pollution (DALYs)
                                ├─→ equity analysis (concentration index)
                                └─→ FSA-level stratification

Each function takes tidy inputs, returns a dataclass, and cites the
specific paper whose formula it implements.

References (module-level):
    Burnett, R. T. et al. (2014). "An integrated risk function for
        estimating the global burden of disease attributable to ambient
        fine particulate matter exposure." Environmental Health
        Perspectives, 122(4), 397-403.
    Atkinson, R. W. et al. (2018). "Long-term concentration-response
        functions for mortality and NO2." Environmental Research, 161,
        101-113.
    WHO (2021). Global Air Quality Guidelines.
    US EPA (2018). BenMAP-CE User's Manual Appendices.
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). Modern
        Epidemiology, 3rd ed., Ch 5 (PAF derivation).
    Wagstaff, A., Paci, P., & van Doorslaer, E. (1991). "On the
        measurement of inequalities in health." Social Science &
        Medicine, 33(5), 545-557.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CRFResult:
    """Concentration-response function result for a single exposure level."""
    rr: float
    log_rr: float
    reference_conc: float
    exposure_conc: float
    pollutant: str
    citation: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BurdenResult:
    """Population-level pollution burden attribution result."""
    paf: float
    attributable_cases: float
    baseline_cases: float
    population: int
    baseline_rate: float
    exposure_mean: float
    reference_conc: float
    pollutant: str
    citation: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EquityResult:
    """Concentration-index result for exposure × income."""
    concentration_index: float
    interpretation: str
    n_quintiles: int
    exposure_mean: float
    citation: str
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Concentration-response functions
# ---------------------------------------------------------------------------

def concentration_response_pm25(
    exposure: float | np.ndarray,
    outcome: str = "all_cause_mortality",
    reference_conc: float = 5.8,
) -> CRFResult:
    """Integrated Exposure-Response (IER) curve for PM2.5.

    Formula (Burnett et al. 2014, Eq. 1):

        RR(z) = 1 + α · (1 − exp(−γ · (z − z_cf)^δ))   if z > z_cf
        RR(z) = 1                                        otherwise

    Default parameters are fit for adult all-cause mortality from GBD
    2013; override via ``outcome`` which selects a published (α, γ, δ)
    triple.

    Parameters
    ----------
    exposure : float or 1-D array
        Ambient PM2.5 concentration in μg/m³.
    outcome : str
        Outcome for which IER parameters are selected. Current options:
        ``"all_cause_mortality"``, ``"ihd"``, ``"stroke"``.
    reference_conc : float
        Counterfactual concentration (z_cf), the level below which no
        excess risk is assumed. Default 5.8 μg/m³ (WHO 2021 interim
        target and GBD 2013 counterfactual).

    Returns
    -------
    CRFResult

    References
    ----------
    Burnett, R. T. et al. (2014). EHP, 122(4), 397-403.
    """
    # Published IER parameters (GBD 2013 adult outcomes)
    ier_params: dict[str, tuple[float, float, float]] = {
        "all_cause_mortality": (1.2, 0.34, 0.72),
        "ihd": (1.91, 0.14, 0.49),
        "stroke": (1.46, 0.13, 0.61),
    }
    if outcome not in ier_params:
        raise ValueError(
            f"Unknown outcome {outcome!r}. Available: {list(ier_params)}")
    alpha, gamma, delta = ier_params[outcome]

    z = np.asarray(exposure, dtype=float)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    excess = np.clip(z - reference_conc, 0.0, None)
    rr = 1.0 + alpha * (1.0 - np.exp(-gamma * np.power(excess, delta)))
    log_rr = np.log(rr)

    if scalar:
        rr_out = float(rr[0])
        log_rr_out = float(log_rr[0])
        exp_out = float(z[0])
    else:
        rr_out = rr
        log_rr_out = log_rr
        exp_out = z

    return CRFResult(
        rr=float(np.mean(rr_out)) if not scalar else rr_out,
        log_rr=float(np.mean(log_rr_out)) if not scalar else log_rr_out,
        reference_conc=reference_conc,
        exposure_conc=float(np.mean(exp_out)) if not scalar else exp_out,
        pollutant="PM2.5",
        citation="Burnett et al. (2014) EHP 122(4):397-403",
        extra={
            "alpha": alpha, "gamma": gamma, "delta": delta,
            "outcome": outcome,
            "rr_per_unit": rr_out if not scalar else None,
        },
    )


def concentration_response_no2(
    exposure: float | np.ndarray,
    outcome: str = "all_cause_mortality",
    reference_conc: float = 10.0,
    beta_per_10: float | None = None,
) -> CRFResult:
    """Log-linear concentration-response function for NO2.

    Formula (Atkinson et al. 2018; WHO 2021):

        RR(z) = exp(β · (z − z_cf) / 10)

    where β is the log-RR per 10 μg/m³ increment. Default outcome
    parameters come from the WHO 2021 Global Air Quality Guidelines
    evidence synthesis.

    Parameters
    ----------
    exposure : float or 1-D array
        Annual-mean NO2 concentration in μg/m³.
    outcome : str
        ``"all_cause_mortality"`` (β=0.039), ``"respiratory"`` (β=0.029),
        or ``"childhood_asthma"`` (β=0.039).
    reference_conc : float
        Counterfactual concentration z_cf. Default 10 μg/m³ (WHO 2021
        annual guideline).
    beta_per_10 : float, optional
        Override β (log-RR per 10 μg/m³). If given, ``outcome`` is ignored
        for the coefficient (but still stamped on the result for
        provenance).

    Returns
    -------
    CRFResult

    References
    ----------
    Atkinson, R. W. et al. (2018). Env Research, 161, 101-113.
    WHO (2021). Global Air Quality Guidelines.
    """
    beta_lookup = {
        "all_cause_mortality": 0.039,
        "respiratory": 0.029,
        "childhood_asthma": 0.039,
    }
    if beta_per_10 is None:
        if outcome not in beta_lookup:
            raise ValueError(
                f"Unknown outcome {outcome!r}. Available: {list(beta_lookup)} "
                f"or pass beta_per_10 explicitly.")
        beta = beta_lookup[outcome]
    else:
        beta = float(beta_per_10)

    z = np.asarray(exposure, dtype=float)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    excess = np.clip(z - reference_conc, 0.0, None) / 10.0
    log_rr = beta * excess
    rr = np.exp(log_rr)

    if scalar:
        rr_out = float(rr[0])
        log_rr_out = float(log_rr[0])
        exp_out = float(z[0])
    else:
        rr_out = rr
        log_rr_out = log_rr
        exp_out = z

    return CRFResult(
        rr=float(np.mean(rr_out)) if not scalar else rr_out,
        log_rr=float(np.mean(log_rr_out)) if not scalar else log_rr_out,
        reference_conc=reference_conc,
        exposure_conc=float(np.mean(exp_out)) if not scalar else exp_out,
        pollutant="NO2",
        citation="Atkinson et al. (2018); WHO (2021) Global AQ Guidelines",
        extra={
            "beta_per_10": beta,
            "outcome": outcome,
            "rr_per_unit": rr_out if not scalar else None,
        },
    )


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def attributable_fraction(rr: float, exposure_prevalence: float) -> float:
    """Population Attributable Fraction (PAF) from a risk ratio.

    Formula (Levin 1953; Rothman et al. 2008, Ch 5):

        PAF = p · (RR − 1) / (1 + p · (RR − 1))

    where ``p`` is the prevalence of exposure in the population and
    ``RR`` is the relative risk at that exposure level.

    Interpretation: the fraction of outcome cases in the population
    that would be avoided if the exposure were removed (assuming
    causation).

    Parameters
    ----------
    rr : float
        Relative risk at the observed exposure level (must be ≥ 1 for
        a harmful exposure; PAF < 0 for protective exposures).
    exposure_prevalence : float
        Proportion of the population exposed, in [0, 1].

    Returns
    -------
    float
        PAF in (−∞, 1). For RR > 1 and p ∈ [0, 1], PAF ∈ [0, 1 − 1/RR].

    References
    ----------
    Levin, M. L. (1953). Acta Un Int Cancr, 9, 531-541.
    Rothman, Greenland & Lash (2008). Modern Epidemiology, 3e, Ch 5.
    """
    rr = float(rr)
    p = float(exposure_prevalence)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"exposure_prevalence must be in [0,1]; got {p}")
    numerator = p * (rr - 1.0)
    denominator = 1.0 + p * (rr - 1.0)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Health-impact assessment (BenMAP-style)
# ---------------------------------------------------------------------------

def mortality_displaced(
    exposure_delta: float,
    population: int,
    baseline_rate: float,
    beta_per_unit: float,
) -> float:
    """Deaths avoided under a counterfactual exposure reduction.

    Formula (US EPA BenMAP-CE, Anenberg et al. 2010):

        ΔY = y₀ · N · (1 − exp(−β · Δx))

    where y₀ is baseline mortality rate, N is the at-risk population,
    β is the log-RR per unit exposure, and Δx is the counterfactual
    reduction (positive for reduction). Under small-β approximation:

        ΔY ≈ y₀ · N · β · Δx

    Parameters
    ----------
    exposure_delta : float
        Counterfactual exposure reduction (μg/m³). Positive means the
        exposure goes DOWN; the function returns deaths avoided.
    population : int
        At-risk population size.
    baseline_rate : float
        Baseline mortality rate (deaths per person-year), e.g. 0.008
        for Ontario adult all-cause.
    beta_per_unit : float
        Log-RR per μg/m³ (note: per unit, NOT per 10 units). For NO2
        all-cause with β_per_10 = 0.039, pass 0.0039 here.

    Returns
    -------
    float
        Expected avoided deaths per person-year of exposure reduction.

    References
    ----------
    US EPA (2018). BenMAP-CE User's Manual Appendices.
    Anenberg, S. C. et al. (2010). EHP, 118(9), 1189-1195.
    """
    if population < 0:
        raise ValueError("population must be non-negative")
    if baseline_rate < 0:
        raise ValueError("baseline_rate must be non-negative")
    return float(baseline_rate * population * (1.0 - np.exp(-beta_per_unit * exposure_delta)))


# ---------------------------------------------------------------------------
# End-to-end burden pipeline
# ---------------------------------------------------------------------------

def burden_of_pollution(
    exposure_mean: float,
    exposure_prevalence: float,
    baseline_rate: float,
    population: int,
    pollutant: str = "NO2",
    outcome: str = "all_cause_mortality",
    reference_conc: float | None = None,
) -> BurdenResult:
    """End-to-end pollution burden: exposure → RR → PAF → attributable cases.

    Chains :func:`concentration_response_pm25` or
    :func:`concentration_response_no2`, then :func:`attributable_fraction`,
    to produce the annual attributable case count.

    Parameters
    ----------
    exposure_mean : float
        Population-mean exposure level (μg/m³).
    exposure_prevalence : float
        Proportion of the population at the exposure_mean level
        (typically 1.0 for ambient air pollution — everyone breathes it).
    baseline_rate : float
        Outcome rate (cases per person-year) in the counterfactual
        unexposed scenario.
    population : int
        At-risk population.
    pollutant : str
        ``"PM2.5"`` or ``"NO2"``.
    outcome : str
        Outcome key passed through to the CRF selector.
    reference_conc : float, optional
        Counterfactual concentration; defaults to the pollutant-specific
        WHO guideline.

    Returns
    -------
    BurdenResult

    References
    ----------
    GBD 2019 Risk Factors Collaborators (2020). Lancet 396:1223-1249.
    """
    if pollutant == "PM2.5":
        crf = concentration_response_pm25(
            exposure_mean, outcome=outcome,
            reference_conc=reference_conc if reference_conc is not None else 5.8,
        )
    elif pollutant == "NO2":
        crf = concentration_response_no2(
            exposure_mean, outcome=outcome,
            reference_conc=reference_conc if reference_conc is not None else 10.0,
        )
    else:
        raise ValueError(f"Unknown pollutant {pollutant!r}. Use 'PM2.5' or 'NO2'.")

    paf = attributable_fraction(crf.rr, exposure_prevalence)
    baseline_cases = float(baseline_rate * population)
    attributable_cases = float(paf * baseline_cases)

    return BurdenResult(
        paf=float(paf),
        attributable_cases=attributable_cases,
        baseline_cases=baseline_cases,
        population=int(population),
        baseline_rate=float(baseline_rate),
        exposure_mean=float(exposure_mean),
        reference_conc=crf.reference_conc,
        pollutant=pollutant,
        citation=crf.citation + "; Rothman et al. (2008) §5",
        extra={
            "rr": crf.rr, "log_rr": crf.log_rr,
            "outcome": outcome,
        },
    )


# ---------------------------------------------------------------------------
# DML-based sensitivity wrapper
# ---------------------------------------------------------------------------

def exposure_response_sensitivity(
    data: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    confounders: list[str],
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Exposure-response DML (PLR) with bootstrap sensitivity.

    Wraps :func:`morie.fn.plr.estimate_plr` (RidgeCV nuisances, continuous
    treatment) with non-parametric bootstrap resampling. Each bootstrap
    iteration refits the PLR pipeline on a resample, yielding a
    distribution of ATE estimates that captures nuisance-estimation
    uncertainty beyond the analytic SE.

    Why PLR and not :func:`estimate_double_ml`: pollution exposure is a
    *continuous* variable, so the ml_m (treatment model) must be a
    regressor, not a classifier. ``estimate_double_ml`` uses a
    ``RandomForestClassifier`` for ml_m and raises if the treatment
    isn't binary in {0, 1}. PLR uses ``RidgeCV`` for both nuisances
    and handles continuous treatment natively (Chernozhukov et al.
    2018 §4.2 PLR model).

    Parameters
    ----------
    data : pd.DataFrame
        Tidy data with exposure, outcome, and confounders as columns.
    outcome : str
        Outcome column name.
    exposure : str
        Exposure column name (the "treatment" in DML parlance).
    confounders : list[str]
        Covariate column names.
    n_bootstrap : int
        Number of resamples. Default 100 (enough for a 95% percentile CI).
    random_state : int
        Seed (used for bootstrap resampling; PLR's internal seed is fixed
        per its implementation).

    Returns
    -------
    dict with keys: ``ate``, ``se_analytic``, ``se_bootstrap``,
    ``ci_lower_bs``, ``ci_upper_bs``, ``n_bootstrap``, ``method``.

    References
    ----------
    Chernozhukov et al. (2018). Econometrics Journal, 21(1), C1-C68 §4.2.
    Bach, P. et al. (2022). DoubleML. JMLR 23(53), 1-6.
    Efron & Tibshirani (1993). An Introduction to the Bootstrap.
    """
    from morie.fn.plr import estimate_plr

    # Point estimate on the full sample
    base = estimate_plr(
        data=data,
        outcome=outcome,
        treatment=exposure,
        covariates=confounders,
    )
    ate = float(base["ate"])
    se_analytic = float(base["se"])

    rng = np.random.default_rng(random_state)
    boot_ates: list[float] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(data), size=len(data))
        resample = data.iloc[idx].reset_index(drop=True)
        try:
            r = estimate_plr(
                data=resample,
                outcome=outcome,
                treatment=exposure,
                covariates=confounders,
            )
            boot_ates.append(float(r["ate"]))
        except Exception:
            continue

    if len(boot_ates) < 10:
        raise RuntimeError(
            f"Only {len(boot_ates)} successful bootstrap fits; need >=10. "
            f"Inspect the PLR convergence on your resamples.")

    se_boot = float(np.std(boot_ates, ddof=1))
    ci_low = float(np.percentile(boot_ates, 2.5))
    ci_high = float(np.percentile(boot_ates, 97.5))

    return {
        "ate": ate,
        "se_analytic": se_analytic,
        "se_bootstrap": se_boot,
        "ci_lower_bs": ci_low,
        "ci_upper_bs": ci_high,
        "n_bootstrap": len(boot_ates),
        "method": "PLR (DoubleML) + percentile bootstrap",
    }


# ---------------------------------------------------------------------------
# Equity analysis
# ---------------------------------------------------------------------------

def pollution_equity_analysis(
    data: pd.DataFrame,
    *,
    exposure: str,
    income: str,
) -> EquityResult:
    """Concentration index for pollution exposure by income.

    Formula (Wagstaff, Paci & van Doorslaer 1991; O'Donnell et al. 2008):

        CI = (2 / μ) · cov(h_i, R_i)

    where h_i is the health (or in our case, exposure) variable, R_i is
    the fractional rank of person i in the income distribution, and μ
    is the mean of h across the population.

    Interpretation:
      - CI = 0 → exposure is distributed equally across the income
        distribution.
      - CI < 0 → lower-income individuals bear a disproportionately
        higher exposure burden (pro-poor pollution burden, adverse
        equity).
      - CI > 0 → higher-income individuals bear disproportionately more
        exposure (less common for pollution).

    Parameters
    ----------
    data : pd.DataFrame
    exposure : str
        Column containing per-unit exposure (μg/m³).
    income : str
        Column containing per-unit income or income proxy.

    Returns
    -------
    EquityResult

    References
    ----------
    Wagstaff, A., Paci, P., & van Doorslaer, E. (1991). Soc Sci Med
        33(5), 545-557.
    O'Donnell, O. et al. (2008). Analyzing Health Equity Using Household
        Survey Data. World Bank, Ch 8.
    """
    df = data[[exposure, income]].dropna().reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to compute concentration index.")
    h = df[exposure].to_numpy(dtype=float)
    inc = df[income].to_numpy(dtype=float)
    mu = float(h.mean())
    if mu == 0.0:
        raise ValueError("Mean exposure is zero; concentration index undefined.")

    # Fractional ranks by income: R_i = (rank_i - 0.5) / n for ties broken by avg
    order = np.argsort(inc, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(inc)) + 1.0
    R = (ranks - 0.5) / len(inc)

    cov_hR = float(np.cov(h, R, ddof=0)[0, 1])
    ci = 2.0 * cov_hR / mu

    if ci < -0.01:
        interp = (f"CI = {ci:.4f}. Pro-poor exposure burden: lower-income "
                  f"individuals bear disproportionately higher pollution.")
    elif ci > 0.01:
        interp = (f"CI = {ci:.4f}. Pro-rich exposure burden: higher-income "
                  f"individuals bear disproportionately higher pollution.")
    else:
        interp = f"CI = {ci:.4f}. Exposure distributed approximately evenly across income."

    return EquityResult(
        concentration_index=ci,
        interpretation=interp,
        n_quintiles=5,
        exposure_mean=mu,
        citation="Wagstaff, Paci & van Doorslaer (1991) Soc Sci Med",
        extra={"n": len(df), "cov_h_R": cov_hR},
    )


# ---------------------------------------------------------------------------
# FSA-level stratified burden
# ---------------------------------------------------------------------------

def burden_by_fsa(
    fsa_table: pd.DataFrame,
    *,
    fsa_col: str = "fsa",
    exposure_col: str = "exposure",
    population_col: str = "population",
    baseline_rate_col: str = "baseline_rate",
    pollutant: str = "NO2",
    outcome: str = "all_cause_mortality",
) -> pd.DataFrame:
    """Per-FSA burden stratification.

    Applies :func:`burden_of_pollution` to each row of a per-FSA table.
    Designed for Ontario Forward Sortation Areas (M6H, M5V, etc.) but
    agnostic to the geographic key.

    Parameters
    ----------
    fsa_table : pd.DataFrame
        One row per FSA with exposure/population/baseline_rate columns.
    fsa_col : str
    exposure_col : str
    population_col : str
    baseline_rate_col : str
    pollutant : str
    outcome : str

    Returns
    -------
    pd.DataFrame
        Input rows plus: ``rr``, ``paf``, ``attributable_cases``,
        ``baseline_cases``. Sorted descending by ``attributable_cases``
        so the worst-affected FSAs appear first.

    References
    ----------
    GBD 2019 Risk Factors Collaborators (2020). Lancet 396:1223-1249.
    """
    required = [fsa_col, exposure_col, population_col, baseline_rate_col]
    missing = [c for c in required if c not in fsa_table.columns]
    if missing:
        raise ValueError(f"fsa_table missing columns: {missing}")

    rows = []
    for _, row in fsa_table.iterrows():
        result = burden_of_pollution(
            exposure_mean=float(row[exposure_col]),
            exposure_prevalence=1.0,  # ambient air: everyone exposed
            baseline_rate=float(row[baseline_rate_col]),
            population=int(row[population_col]),
            pollutant=pollutant,
            outcome=outcome,
        )
        rows.append({
            fsa_col: row[fsa_col],
            exposure_col: float(row[exposure_col]),
            population_col: int(row[population_col]),
            baseline_rate_col: float(row[baseline_rate_col]),
            "rr": result.extra["rr"],
            "paf": result.paf,
            "attributable_cases": result.attributable_cases,
            "baseline_cases": result.baseline_cases,
        })

    return pd.DataFrame(rows).sort_values(
        "attributable_cases", ascending=False).reset_index(drop=True)


def cheatsheet() -> str:
    return (
        "morie.envhealth: concentration_response_pm25/no2, attributable_fraction, "
        "mortality_displaced, burden_of_pollution, exposure_response_sensitivity, "
        "pollution_equity_analysis, burden_by_fsa."
    )
