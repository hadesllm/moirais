# moirais.fn — function file (hadesllm/moirais)
"""PM₂.₅ attributable-burden via GEMM nonlinear RR (Burnett 2018)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._containers import DescriptiveResult
from .pmgemm import pm_gemm_rr


def pm_gemm_burden(
    concentration_ugm3: float | np.ndarray,
    population: int | np.ndarray,
    baseline_rate: float,
    *,
    outcome: Literal[
        "ncd_lri", "ihd", "stroke", "copd", "lung_cancer", "lri"
    ] = "ncd_lri",
    reference_ugm3: float = 2.4,
    theta: float | None = None,
    alpha: float | None = None,
    mu: float | None = None,
    nu: float | None = None,
) -> DescriptiveResult:
    """Compute PM₂.₅ attributable deaths using the Burnett GEMM curve.

    Composes ``pm_gemm_rr`` (nonlinear RR) with the classical
    population-attributable-fraction formula and a mortality-rate
    scaler to produce attributable deaths per year per unit. Unlike
    ``moirais.envhealth.burden_of_pollution`` which uses a log-linear
    concentration-response, this helper uses the GEMM nonlinear
    curve and saturates appropriately at high concentrations.

    Formula:

    .. math::

        PAF(C) = \\frac{RR_{GEMM}(C) - 1}{RR_{GEMM}(C)}

    .. math::

        \\text{deaths}_{\\text{attrib}} = PAF(C) \\cdot r_0 \\cdot N

    where :math:`r_0` is baseline all-cause (or outcome-specific)
    mortality per person-year and :math:`N` is the at-risk
    population.

    Parameters
    ----------
    concentration_ugm3 : float or array-like
        PM₂.₅ annual mean per unit (µg/m³). Scalar for single
        analysis; array of length N for per-unit loops (FSA list,
        city list, etc.).
    population : int or array-like
        At-risk population. Shape must broadcast against
        ``concentration_ugm3``.
    baseline_rate : float
        Baseline mortality rate (deaths per person-year). For
        NCD+LRI all-adults, ~0.008 in Canada (CIHI); use
        outcome-specific rates when analyzing a single disease.
    outcome : str, default "ncd_lri"
        Which GEMM curve to use — passed through to ``pm_gemm_rr``.
    reference_ugm3 : float, default 2.4
        TMREL (theoretical minimum risk exposure level) per
        Burnett 2018.
    theta, alpha, mu, nu : float, optional
        Override the GEMM shape parameters for sensitivity analysis.

    Returns
    -------
    DescriptiveResult
        value = total attributable deaths/year across all units.
        extra has per-unit RR, PAF, deaths, plus totals.

    Examples
    --------
    Canadian city, annual mean PM₂.₅ = 10 µg/m³, 500k people,
    baseline NCD+LRI mortality 0.008:

    >>> r = pm_gemm_burden(10.0, 500_000, 0.008)
    >>> round(r.value, 1)  # low exposure → modest attribution
    159.1

    High-pollution city, 80 µg/m³, 1M people:

    >>> r = pm_gemm_burden(80.0, 1_000_000, 0.008)
    >>> r.extra["paf"] > 0.3   # PAF saturates around 0.4 via GEMM
    True

    Per-FSA vectorized run across an array of units:

    >>> import numpy as np
    >>> concs = np.array([8.0, 12.0, 15.0])
    >>> pops  = np.array([25_000, 30_000, 20_000])
    >>> r = pm_gemm_burden(concs, pops, 0.008)
    >>> len(r.extra["deaths_per_unit"])
    3

    References
    ----------
    Burnett, R. et al. (2018). Global estimates of mortality
    associated with long-term exposure to outdoor fine particulate
    matter. PNAS 115(38), 9592-9597.

    GBD 2019 Risk Factor Collaborators (2020). Global burden of 87
    risk factors in 204 countries and territories, 1990-2019: a
    systematic analysis for the Global Burden of Disease Study
    2019. Lancet 396(10258), 1223-1249.

    Notes
    -----
    Quote: "The saturation is the lesson." Burnett GEMM was built
    specifically so that Delhi-scale exposures don't yield
    fictional orders-of-magnitude mortality under linear
    extrapolation. This helper carries that saturation through
    to the attributable-deaths numerator.
    """
    if baseline_rate < 0:
        raise ValueError("baseline_rate must be non-negative.")

    C = np.atleast_1d(np.asarray(concentration_ugm3, dtype=float))
    N = np.atleast_1d(np.asarray(population, dtype=float))

    if N.shape != C.shape:
        try:
            N = np.broadcast_to(N, C.shape)
        except ValueError as exc:
            raise ValueError(
                f"population shape {N.shape} does not broadcast to "
                f"concentration shape {C.shape}"
            ) from exc

    if np.any(N < 0):
        raise ValueError("population must be non-negative.")

    # Delegate RR to pmgemm (handles all parameter sanity).
    rr_result = pm_gemm_rr(
        C,
        outcome=outcome,
        reference_ugm3=reference_ugm3,
        theta=theta, alpha=alpha, mu=mu, nu=nu,
    )
    rr_arr = np.atleast_1d(np.asarray(
        rr_result.extra["rr"], dtype=float,
    ))

    # PAF = (RR - 1) / RR.  When RR=1, PAF=0 exactly.
    paf = np.where(rr_arr > 1.0, (rr_arr - 1.0) / rr_arr, 0.0)
    deaths_per_unit = paf * baseline_rate * N

    total_deaths = float(deaths_per_unit.sum())
    total_pop = float(N.sum())

    extra: dict = {
        "rr":                rr_arr.tolist() if rr_arr.size > 1 else float(rr_arr.item()),
        "paf":               paf.tolist() if paf.size > 1 else float(paf.item()),
        "deaths_per_unit":   deaths_per_unit.tolist() if deaths_per_unit.size > 1
                             else float(deaths_per_unit.item()),
        "total_deaths":      total_deaths,
        "total_population":  total_pop,
        "outcome":           outcome,
        "baseline_rate":     baseline_rate,
        "reference_ugm3":    reference_ugm3,
        "source":            "Burnett 2018 PNAS GEMM + Levin 1953 PAF",
    }
    if total_pop > 0:
        extra["crude_attributable_rate_per_100k"] = (
            total_deaths / total_pop * 100_000
        )

    return DescriptiveResult(
        name="pm_gemm_burden",
        value=total_deaths,
        extra=extra,
    )


pmgbrd = pm_gemm_burden


def cheatsheet() -> str:
    return "pmgbrd(C_ugm3, pop, rate, outcome='ncd_lri') -> PM2.5 attrib deaths via GEMM."
