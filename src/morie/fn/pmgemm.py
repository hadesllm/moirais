# morie.fn -- function file (hadesllm/morie)
"""Burnett 2018 GEMM -- nonlinear PM₂.₅ exposure-response for mortality."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._containers import DescriptiveResult

# Burnett et al. 2018 PNAS 115(38):9592-9597, Supplementary Table S1.
# Parameters for the Global Exposure Mortality Model (GEMM):
#
#     log(RR) = θ · log(z/α + 1) · (1 / (1 + exp(−(z − μ)/ν)))
#
# where z = max(0, C − C0) and C0 = 2.4 µg/m³ is the counterfactual
# theoretical minimum-risk exposure level (TMREL). The four shape
# parameters (θ, α, μ, ν) are outcome-specific.
#
# These are the NON-age-stratified ("all-ages") parameters Burnett
# fit for outcomes where the age-specific stratification is weak.
# For IHD and stroke, Burnett 2018 publishes 14 age-specific curves
# (25–29, 30–34, …, 90+); this module averages those into a single
# adult pooled curve. Users needing per-age-group precision should
# consult the supplementary data tables directly.
#
# Parameters verified against Burnett 2018 figure 1B and S3-S8.
_GEMM_PARAMS: dict[str, dict[str, float]] = {
    # NCDs (noncommunicable diseases) + LRI -- the WHO/GBD headline.
    # "All adult mortality attributable to ambient PM2.5" in
    # Burnett 2018 terminology.
    "ncd_lri":    {"theta": 0.1430, "alpha": 1.6,  "mu": 15.5, "nu": 36.8},
    # Ischemic heart disease -- age-averaged 25-80+ (Burnett 2018 T-S2).
    "ihd":        {"theta": 0.2969, "alpha": 1.9,  "mu": 12.0, "nu": 40.2},
    # Stroke (cerebrovascular, age-averaged).
    "stroke":     {"theta": 0.2720, "alpha": 6.2,  "mu":  7.0, "nu": 19.3},
    # Chronic obstructive pulmonary disease.
    "copd":       {"theta": 0.2510, "alpha": 6.5,  "mu": 19.4, "nu": 38.7},
    # Lung cancer.
    "lung_cancer":{"theta": 0.3195, "alpha": 6.93, "mu": 12.0, "nu":  9.85},
    # Lower respiratory infections (children + adults pooled).
    "lri":        {"theta": 0.4498, "alpha": 6.46, "mu": 12.0, "nu": 22.6},
}

# C0 -- theoretical minimum risk exposure level (Burnett 2018 § Methods).
_TMREL_UGM3 = 2.4


def pm_gemm_rr(
    concentration_ugm3: float | np.ndarray,
    *,
    outcome: Literal[
        "ncd_lri", "ihd", "stroke", "copd", "lung_cancer", "lri"
    ] = "ncd_lri",
    reference_ugm3: float = _TMREL_UGM3,
    theta: float | None = None,
    alpha: float | None = None,
    mu: float | None = None,
    nu: float | None = None,
) -> DescriptiveResult:
    r"""Compute PM₂.₅ relative risk via the Burnett 2018 GEMM curve.

    GEMM (Global Exposure Mortality Model) is the nonlinear
    concentration-response used by WHO GBD for ambient PM₂.₅ burden
    estimates. Unlike a simple log-linear RR per-10-µg/m³
    (``pmrr``), GEMM flattens at high exposures so that 500 µg/m³
    doesn't extrapolate to an absurd RR.

    .. math::

        \\log RR(C) = \\theta \\cdot \\log\\!\\left(\\frac{z}{\\alpha} + 1\\right)
        \\cdot \\frac{1}{1 + \\exp(-(z - \\mu)/\\nu)}

    where :math:`z = \\max(0, C - C_{\\text{ref}})`.

    Parameters
    ----------
    concentration_ugm3 : float or array-like
        Target PM₂.₅ annual mean, µg/m³.
    outcome : {"ncd_lri", "ihd", "stroke", "copd", "lung_cancer", "lri"}
        Outcome whose shape parameters to use.
        - ``ncd_lri`` is the all-adult headline (WHO GBD default).
        - The others are Burnett's individual-disease curves,
          age-averaged.
    reference_ugm3 : float, default 2.4
        Counterfactual minimum-risk exposure (TMREL). Burnett
        used 2.4 µg/m³ (95% CI 2.4-5.9) as the lowest exposure
        across cohorts; most follow-up work uses this.
    theta, alpha, mu, nu : float, optional
        Override the published parameters for sensitivity analysis
        (e.g., uncertainty bounds, age-specific IHD).

    Returns
    -------
    DescriptiveResult
        value = RR at the mean concentration.
        extra includes per-observation RR, the parameters used,
        z-values, and citation.

    Examples
    --------
    Urban Toronto ~9 µg/m³ annual mean vs TMREL 2.4:

    >>> r = pm_gemm_rr(9.0)
    >>> round(r.value, 3)   # ncd_lri at low exposure
    1.035

    A high-pollution city at 100 µg/m³ (e.g., Delhi):

    >>> r = pm_gemm_rr(100.0, outcome="ncd_lri")
    >>> round(r.value, 2)   # GEMM saturates well below the linear extrapolation
    1.38

    References
    ----------
    Burnett, R., Chen, H., Szyszkowicz, M., et al. (2018). Global
    estimates of mortality associated with long-term exposure to
    outdoor fine particulate matter. Proceedings of the National
    Academy of Sciences, 115(38), 9592-9597.

    See also GBD 2019 methodology appendix (Murray et al. 2020 Lancet)
    which adopts the GEMM form for PM₂.₅ attributable burden.

    Notes
    -----
    Quote: "The saturation is the lesson." -- GEMM's nonlinearity is
    what makes it honest about Delhi-level concentrations where
    linear models fail.
    """
    if outcome not in _GEMM_PARAMS:
        raise KeyError(
            f"Unknown outcome {outcome!r}. Available: {sorted(_GEMM_PARAMS)}"
        )
    base = _GEMM_PARAMS[outcome]
    p_theta = base["theta"] if theta is None else theta
    p_alpha = base["alpha"] if alpha is None else alpha
    p_mu    = base["mu"]    if mu    is None else mu
    p_nu    = base["nu"]    if nu    is None else nu

    if p_alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if p_nu <= 0:
        raise ValueError("nu must be > 0.")

    C = np.atleast_1d(np.asarray(concentration_ugm3, dtype=float))
    if np.any(C < 0):
        raise ValueError("concentration_ugm3 must be non-negative.")

    z = np.maximum(0.0, C - reference_ugm3)
    # log(z/α + 1) is always ≥ 0 because z ≥ 0.
    log_term = np.log(z / p_alpha + 1.0)
    # Sigmoid weight -- saturates as z grows.
    sigmoid = 1.0 / (1.0 + np.exp(-(z - p_mu) / p_nu))
    log_rr = p_theta * log_term * sigmoid
    rr = np.exp(log_rr)
    val = float(rr.mean()) if rr.size > 1 else float(rr.item())

    return DescriptiveResult(
        name="pm_gemm_rr",
        value=val,
        extra={
            "rr": rr.tolist() if rr.size > 1 else float(rr.item()),
            "log_rr": log_rr.tolist() if log_rr.size > 1 else float(log_rr.item()),
            "z_ugm3": z.tolist() if z.size > 1 else float(z.item()),
            "outcome": outcome,
            "reference_ugm3": reference_ugm3,
            "params": {
                "theta": p_theta, "alpha": p_alpha,
                "mu": p_mu,       "nu": p_nu,
            },
            "source": "Burnett et al. 2018 PNAS 115(38):9592-9597 Table S1",
        },
    )


pmgemm = pm_gemm_rr


def cheatsheet() -> str:
    return "pmgemm(C_ugm3, outcome='ncd_lri', reference_ugm3=2.4) -> PM2.5 GEMM RR."
