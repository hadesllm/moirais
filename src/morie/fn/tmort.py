"""Temperature-mortality V-curve via Minimum Mortality Temperature."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# Gasparrini et al. 2015 Lancet 386:369-375 multi-country study
# ("Mortality risk attributable to high and low ambient temperature")
# documented J-shaped / V-shaped temperature-mortality curves across
# 384 locations in 13 countries. The minimum-mortality temperature
# (MMT) falls typically in the 80-90th percentile of the local
# temperature distribution (i.e., warm but not extreme).
#
# A workable parametric approximation is:
#
#     ln(RR(T)) = β_hot * max(0, T - MMT)       (heat arm)
#               + β_cold * max(0, MMT - T)      (cold arm)
#
# where β_hot and β_cold are per-°C log-RR slopes. Gasparrini's
# pooled estimates (averaged across temperate and cold-climate
# cities):
#
#     β_hot  ≈ 0.012 per °C above MMT (1.2% excess per °C)
#     β_cold ≈ 0.003 per °C below MMT (0.3% excess per °C)
#
# These are central estimates; region-specific slopes vary. Users
# who have local slopes should pass them explicitly.
_GASPARRINI_POOLED_BETA_HOT = 0.012
_GASPARRINI_POOLED_BETA_COLD = 0.003


def temperature_mortality_vcurve(
    T_C: float | np.ndarray,
    mmt_C: float,
    *,
    beta_hot: float = _GASPARRINI_POOLED_BETA_HOT,
    beta_cold: float = _GASPARRINI_POOLED_BETA_COLD,
) -> DescriptiveResult:
    r"""Compute daily mortality relative risk from temperature via V-curve.

    Implements the minimum-mortality-temperature framework from
    Gasparrini et al. 2015 Lancet: mortality is lowest at the MMT
    (location-specific, usually 80–90th percentile of local temp),
    and rises exponentially with deviation above OR below MMT.
    Above-MMT slope typically 3–4× larger than below-MMT (heat kills
    faster than cold per-°C).

    .. math::

        RR(T) = \\exp\\bigl(
            \\beta_{hot} \\cdot \\max(0, T - MMT) +
            \\beta_{cold} \\cdot \\max(0, MMT - T)
        \\bigr)

    Parameters
    ----------
    T_C : float or array-like
        Daily mean temperature, °C.
    mmt_C : float
        Minimum Mortality Temperature for the location, °C. Common
        values: 18-22°C in temperate cities (Gasparrini Table S4).
    beta_hot : float, default 0.012
        Log-RR slope per °C above MMT. 0.012 = 1.2% excess mortality
        per °C above MMT (Gasparrini pooled).
    beta_cold : float, default 0.003
        Log-RR slope per °C below MMT. 0.003 = 0.3% excess mortality
        per °C below MMT.

    Returns
    -------
    DescriptiveResult
        value = RR at mean temperature (dimensionless, >= 1).
        extra has per-observation RR, MMT, slopes, and mean attributable
        fraction AF = (RR - 1) / RR.

    Examples
    --------
    Toronto, MMT ≈ 22°C. Mortality RR on a 32°C summer day:

    >>> r = temperature_mortality_vcurve(32.0, mmt_C=22.0)
    >>> round(r.value, 3)   # exp(0.012 * 10) = 1.127
    1.127

    And on a -15°C winter day:

    >>> r = temperature_mortality_vcurve(-15.0, mmt_C=22.0)
    >>> round(r.value, 3)   # exp(0.003 * 37) = 1.117
    1.117

    References
    ----------
    Gasparrini, A., Guo, Y., Hashizume, M., Lavigne, E., Zanobetti, A.,
    Schwartz, J., ... Armstrong, B. (2015). Mortality risk attributable
    to high and low ambient temperature: a multicountry observational
    study. The Lancet, 386(9991), 369-375.

    Notes
    -----
    Quote: "The cold and the hot both know your name, and the cold
    knows it longer."

    Assumes no adaptation / acclimatization — the V-curve encodes
    acute exposure risk. For long-term climate-attribution, combine
    with heat-wave detection (``heatwv``) and demographic weighting.
    """
    T = np.atleast_1d(np.asarray(T_C, dtype=float))
    mmt = float(mmt_C)
    if beta_hot < 0 or beta_cold < 0:
        raise ValueError("beta_hot and beta_cold must be non-negative.")

    above = np.maximum(0.0, T - mmt)
    below = np.maximum(0.0, mmt - T)
    log_rr = beta_hot * above + beta_cold * below
    rr = np.exp(log_rr)

    val = float(rr.mean()) if rr.size > 1 else float(rr.item())
    af = (rr - 1.0) / rr   # attributable fraction per observation

    return DescriptiveResult(
        name="temperature_mortality_vcurve",
        value=val,
        extra={
            "rr": rr.tolist() if rr.size > 1 else float(rr.item()),
            "attributable_fraction": (af.tolist() if af.size > 1
                                       else float(af.item())),
            "mmt_C": mmt,
            "beta_hot": beta_hot,
            "beta_cold": beta_cold,
            "mean_rr": val,
            "mean_af": float(af.mean()) if af.size > 1 else float(af.item()),
            "source": "Gasparrini et al. 2015 Lancet 386:369-375",
        },
    )


tmort = temperature_mortality_vcurve


def cheatsheet() -> str:
    return "tmort(T, mmt_C=22) -> Gasparrini V-curve daily-mortality RR."
