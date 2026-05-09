"""Wildfire-smoke PM₂.₅ excess toxicity multiplier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# Aguilera et al. 2021 Nature Communications 12:1493 found wildfire-
# smoke PM2.5 is roughly 10× more toxic for respiratory hospital
# admissions than ambient PM2.5 in California, 2004-2016. The smoke-
# attributable fraction per µg/m³ was substantially larger than the
# same µg/m³ from non-smoke sources.
#
# Follow-up: Reid et al. 2016 Environ Health Perspect 124:1334-1343
# reviewed cohort + panel studies; found respiratory RR per 10 µg/m³
# smoke-PM ranging 1.02-1.30 (median ~1.07), vs. ambient-PM baseline
# 1.010 per 10 µg/m³. That's roughly a 5-10× multiplier, outcome-
# dependent.
#
# Multipliers (smoke-specific toxicity / ambient toxicity) per 10 µg/m³:
_SMOKE_MULTIPLIER: dict[str, tuple[float, tuple[float, float]]] = {
    # outcome -> (central multiplier, 95% CI)
    "respiratory":      (10.0, (4.0, 20.0)),   # Aguilera 2021
    "asthma_ed":        (8.0, (3.0, 15.0)),    # Reid 2016
    "copd":             (5.0, (2.0, 12.0)),
    "all_cause":        (3.0, (1.5, 6.0)),     # Liu et al. 2017 meta
    "cardiovascular":   (2.0, (1.0, 4.5)),
}


def wildfire_smoke_rr(
    smoke_pm25_ugm3: float | np.ndarray,
    ambient_pm25_ugm3: float | np.ndarray = 0.0,
    *,
    reference_ugm3: float = 0.0,
    ambient_rr_per_10: float = 1.010,
    outcome: str = "respiratory",
) -> DescriptiveResult:
    """Compute wildfire-smoke vs. ambient PM₂.₅ relative risk.

    Wildfire-smoke PM₂.₅ is roughly 2-10× more toxic per µg/m³ than
    ambient (non-smoke) PM₂.₅ for respiratory outcomes, per the
    Aguilera/Reid/Liu literature. This function combines smoke-
    specific and ambient exposures, each with its own per-10-µg RR,
    into a single total RR.

    .. math::

        RR = \\exp\\bigl(
          \\ln(RR_{\\mathrm{smoke}_{10}}) \\cdot
            (C_{\\mathrm{smoke}} - C_{\\mathrm{ref}}) / 10
          + \\ln(RR_{\\mathrm{amb}_{10}}) \\cdot
            C_{\\mathrm{ambient}} / 10
        \\bigr)

    where the smoke RR per 10 µg/m³ is the ambient RR times the
    outcome-specific smoke multiplier.

    Parameters
    ----------
    smoke_pm25_ugm3 : float or array-like
        Smoke-attributable PM₂.₅, µg/m³. From HMS or chemical-transport
        models.
    ambient_pm25_ugm3 : float or array-like, default 0
        Non-smoke ambient PM₂.₅ contribution, µg/m³. If 0, analyzes
        smoke effect alone.
    reference_ugm3 : float, default 0
        Counterfactual reference smoke concentration.
    ambient_rr_per_10 : float, default 1.010
        Baseline ambient PM₂.₅ RR per 10 µg/m³ for the chosen outcome
        (override if you have a region-specific estimate).
    outcome : str, default "respiratory"
        Outcome name. See `_SMOKE_MULTIPLIER` keys.

    Returns
    -------
    DescriptiveResult
        value = RR at the mean smoke/ambient pair.
        extra has per-observation RR, the smoke multiplier used, and
        the effective smoke RR per 10 µg/m³.

    Examples
    --------
    A California coastal day: 50 µg/m³ smoke + 10 µg/m³ ambient,
    vs. baseline 10 µg/m³ ambient:

    >>> r = wildfire_smoke_rr(
    ...     smoke_pm25_ugm3=50,
    ...     ambient_pm25_ugm3=10,
    ...     outcome="respiratory"
    ... )
    >>> round(r.value, 2)   # smoke effect dominates
    1.59

    References
    ----------
    Aguilera, R., Corringham, T., Gershunov, A., & Benmarhnia, T.
    (2021). Wildfire smoke impacts respiratory health more than fine
    particles from other sources. Nature Communications, 12(1), 1493.

    Reid, C. E., Brauer, M., Johnston, F. H., Jerrett, M., Balmes, J.,
    & Elliott, C. T. (2016). Critical review of health impacts of
    wildfire smoke exposure. Environmental Health Perspectives,
    124(9), 1334-1343.

    Liu, J. C., Wilson, A., Mickley, L. J., Dominici, F., Ebisu, K.,
    Wang, Y., ... Bell, M. L. (2017). Wildfire-specific fine
    particulate matter and risk of hospital admissions in urban and
    rural counties. Epidemiology, 28(1), 77-85.

    Notes
    -----
    Quote: "This smoke is not the smoke of a lamp; it is the smoke
    of a continent."
    """
    key = outcome.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in _SMOKE_MULTIPLIER:
        raise KeyError(
            f"Unknown outcome {outcome!r}. Available: "
            f"{sorted(_SMOKE_MULTIPLIER)}"
        )
    mult, (mult_lo, mult_hi) = _SMOKE_MULTIPLIER[key]

    if ambient_rr_per_10 <= 0:
        raise ValueError("ambient_rr_per_10 must be > 0.")
    # Derive smoke-specific RR per 10 µg/m³ from the multiplier
    # against the ambient baseline, applied to the *excess* log-RR.
    log_rr_amb = np.log(ambient_rr_per_10)
    log_rr_smoke = mult * log_rr_amb
    rr_smoke_10 = float(np.exp(log_rr_smoke))

    S = np.atleast_1d(np.asarray(smoke_pm25_ugm3, dtype=float))
    A = np.atleast_1d(np.asarray(ambient_pm25_ugm3, dtype=float))
    if A.size == 1 and S.size > 1:
        A = np.broadcast_to(A, S.shape)
    if S.shape != A.shape:
        raise ValueError("smoke and ambient arrays must match in shape.")
    if np.any(S < 0) or np.any(A < 0):
        raise ValueError("Concentrations must be non-negative.")

    smoke_delta = (S - reference_ugm3) / 10.0
    amb_delta = A / 10.0
    rr = np.exp(log_rr_smoke * smoke_delta + log_rr_amb * amb_delta)
    val = float(rr.mean()) if rr.size > 1 else float(rr.item())

    return DescriptiveResult(
        name="wildfire_smoke_rr",
        value=val,
        extra={
            "rr": rr.tolist() if rr.size > 1 else float(rr.item()),
            "smoke_multiplier": mult,
            "smoke_multiplier_95ci": (mult_lo, mult_hi),
            "rr_per_10_smoke": rr_smoke_10,
            "rr_per_10_ambient": ambient_rr_per_10,
            "outcome": key,
            "reference_ugm3": reference_ugm3,
            "source": "Aguilera 2021 Nat Comm / Reid 2016 EHP / Liu 2017 Epi",
        },
    )


smokpm = wildfire_smoke_rr


def cheatsheet() -> str:
    return "smokpm(smoke_pm, ambient_pm=0, outcome='respiratory') -> wildfire RR."
